// Copyright 2014 Brian Koropoff
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

# STM for Rust

`libstim` is an (incomplete) implementation of software transactional
memory based roughly on the design used in the Haskell STM library.

For examples, look at the tests at the end of the source.

# Caveats

This library requires a hacked-up version of libstd that allows
a task to recover from unwinding.  This is used to abort and restart
transactions and recover from transient errors due to the program
having an inconsistent view of variables.  It also makes the interface
a lot easier to use since client applications don't need to manually
propagate transaction failures.

https://github.com/bkoropoff/rust/tree/unwind-hacks

# References

## Paper on Haskell implementation

http://research.microsoft.com/en-us/um/people/simonpj/papers/stm/stm.pdf

*/

#![crate_id="stim#0.1"]
#![feature(globs,phase)]

extern crate collections;
#[phase(syntax,link)]
extern crate log;

use std::sync::atomics::{AtomicUint,Relaxed,Acquire,Release,fence};
use std::intrinsics::{atomic_xadd_relaxed};
use std::cast::{transmute,forget};
use std::ptr::{read};
use std::ty::{Unsafe};
use std::kinds::marker::{NoShare,NoSend,ContravariantLifetime};
use std::mem::{drop};
use std::cell::{Cell};
// These two functions are hacks added to my branch
use std::rt::begin_unwind_quiet;
use std::rt::task::try_catch;
use std::any::{AnyRefExt};
use std::local_data;
use collections::hashmap::{HashMap};

static LOCKED_READ: uint = 0;
static LOCKED_COMMIT: uint = 1;
static INITIAL_EPOCH: uint = 2;
static mut GLOBAL_EPOCH: uint = INITIAL_EPOCH + 1;

// Current transaction on this task.  We use *()
// as the type to get around the required 'static
// bound.
local_data_key!(cur_transaction: *())

struct VarBox<T> {
    value: Unsafe<T>,
    refcount: AtomicUint,
    state: AtomicUint
}

impl<T: Send + Clone> VarBox<T> {
}

/// A transaction variable.
///
/// Transaction variables may only be accessed within a transaction
/// created by the `atomically` function.
/// 
/// # Bounds
/// 
/// The contents of a variable must be:
///
/// - `Send`, as it can be passed between different tasks.  
/// - `Clone`, as transactions are executed optimistically
///    by each task working on a local copy of the variable's contents
///    and commiting modifications atomically at the end.
///
/// A variable itself is `Send` but not `Share`.  A variable can be
/// cloned to share access to the value among tasks in a manner analagous
/// to `Arc`.  Notably, this means that a variable satisfies the bounds
/// necessary to be stored in a variable itself.
pub struct Var<T> {
    it: *mut VarBox<T>,
    no_share: NoShare
}

impl<T: Send + Clone> Var<T> {
    // Locks the variable to read its value.
    // This will fail if its current commit epoch is greater
    // than that of the transaction attempting to read it,
    // the reasoning being that the transaction will almost
    // certainly fail to commit since the variable has been
    // concurrently modified by other transactions.
    fn lock_read(&self, epoch: uint) -> Option<uint> {
        let state = unsafe { &(*self.it).state };
        let mut old_val = state.load(Relaxed);
        let mut val;

        // CAS loop
        loop {
            val = old_val;
            while val == LOCKED_READ || val == LOCKED_COMMIT {
                // FIXME: do a better spin than this
                val = state.load(Relaxed);
            }
            // Fail if the variable has been updated
            if val > epoch {
                return None
            }
            // Use Acquire ordering so we see any modifications made
            // to the variable contents by committers.
            old_val = state.compare_and_swap(val, LOCKED_READ, Acquire);
            if old_val == val {
                break;
            }
        }

        Some(val)
    }

    // Unlocks the variable, setting the commit epoch
    fn unlock(&self, epoch: uint) {
        unsafe { (*self.it).state.store(epoch, Release); }
    }

    // Locks the variable for committing.      
    // This will fail if its current commit epoch is greater
    // than that of the transaction attempting to read it,
    // or if it is already locked for commit by another
    // transaction.
    fn lock_commit(&self, epoch: uint) -> Option<uint> {
        let state = unsafe { &(*self.it).state };
        let mut old_val = state.load(Relaxed);
        let mut val;

        // CAS loop
        loop {
            val = old_val;
            while val == LOCKED_READ {
                // FIXME: do a better spin than this
                val = state.load(Relaxed);
            }
            // Fail if the variable is locked by another
            // transaction or has been updated
            if val == LOCKED_COMMIT || val > epoch {
                return None
            }
            old_val = state.compare_and_swap(val, LOCKED_COMMIT, Acquire);
            if old_val == val {
                break;
            }
        }

        Some(val)
    }

    // Verifies that the variable has not been changed since
    // `epoch`.  Note that this can return a false positive
    // since we are using Relaxed ordering, but this just
    // means the transaction will fail to commit later.
    fn verify(&self, epoch: uint) -> bool {
        unsafe { (*self.it).state.load(Relaxed) == epoch }
    }

    // Reads the current value (must have read lock)
    unsafe fn read(&self) -> T {
        (*(*self.it).value.get()).clone()
    }

    // Writes the current value (must have commit lock)
    unsafe fn write(&self, value: T) {
        *(*self.it).value.get() = value;
    }

    /// Create a new transaction variable
    ///
    /// The variable is initialized to `val`
    pub fn new(val: T) -> Var<T> {
        Var {
            it: unsafe { 
                transmute(~VarBox {
                    value: Unsafe::new(val),
                    refcount: AtomicUint::new(1),
                    state: AtomicUint::new(INITIAL_EPOCH)
                })
            },
            no_share: NoShare
        }
    }

    /// Read variable
    ///
    /// Returns a clone the the contents of the variable
    /// as a transaction.
    pub fn get(&self) -> T {
        atomically(|trans| trans.get(self))
    }

    /// Set variable
    ///
    /// Sets the variable to a clone of the specified value
    /// as a transaction.
    pub fn set(&self, val: T) {
        atomically(|trans| trans.set(self, val.clone()))
    }
}

impl<T: Send + Clone> Clone for Var<T> {
    /// Clone variable
    ///
    /// Creates a clone of the variable which points to
    /// the same logical shared contents.  Clones may
    /// be given to other tasks to participate in transactions
    /// involving the variable.
    ///
    /// The contents of the variable are dropped when all clones
    /// are dropped.
    fn clone(&self) -> Var<T> {
        unsafe {
            (*self.it).refcount.fetch_add(1, Relaxed);
        }

        Var { 
            it: self.it,
            no_share: NoShare
        }
    }
}

#[unsafe_destructor]
impl<T: Send + Clone> Drop for Var<T> {
    fn drop(&mut self) {
        unsafe {
            if (*self.it).refcount.fetch_sub(1, Release) == 1 {
                fence(Acquire);
                let _: ~VarBox<T> = transmute(self.it);
            }
        }
    }
}

#[deriving(Eq)]
enum Loan {
    NoLoan,
    ImmLoan,
    MutLoan,
}

// Local copy of a transaction variable in
// a transaction log.  This structure tracks
// the value's borrow state and dirtiness.
struct Value<T> {
    // The value
    value: Unsafe<T>,
    // Is the value dirty (needs to be writen back
    // to the transaction variable)?
    dirty: Cell<bool>,
    // Tracks extant borrows of the value
    loan: Cell<Loan>
}

/// Immutable variable borrow
pub struct Borrow<'a, T> {
    value: *Value<T>,
    no_share: NoShare,
    no_send: NoSend,
    lifetime: ContravariantLifetime<'a>
}

#[unsafe_destructor]
impl<'a, T> Drop for Borrow<'a, T> {
    fn drop(&mut self) {
        unsafe {
            (*self.value).loan.set(NoLoan);
        }
    }
}

impl<'a, T> Deref<T> for Borrow<'a, T> {
    fn deref<'a>(&'a self) -> &'a T {
        unsafe {
            &*(*self.value).value.get()
        }
    }
}

/// Mutable variable borrow
pub struct MutBorrow<'a, T> {
    value: *mut Value<T>,
    no_share: NoShare,
    no_send: NoSend,
    lifetime: ContravariantLifetime<'a>
}

#[unsafe_destructor]
impl<'a, T> Drop for MutBorrow<'a, T> {
    fn drop(&mut self) {
        unsafe {
            (*self.value).loan.set(NoLoan);
        }
    }
}

impl<'a, T> Deref<T> for MutBorrow<'a, T> {
    fn deref<'a>(&'a self) -> &'a T {
        unsafe {
            &*(*self.value).value.get()
        }
    }
}

impl<'a, T> DerefMut<T> for MutBorrow<'a, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe {
            &mut *(*self.value).value.get()
        }
    }
}

impl<T: Send + Clone> Value<T> {
    fn new(val: T) -> Value<T> {
        Value {
            value: Unsafe::new(val),
            dirty: Cell::new(false),
            loan: Cell::new(NoLoan)
        }
    }

    fn is_dirty(&self) -> bool {
        self.dirty.get()
    }

    // Writes value back to variable.  This
    // takes self by value because we move
    // the value to avoid an unneeded clone.
    fn writeback(self, var: &Var<T>) {
        unsafe {
            var.write(read(self.value.get() as *T));
            forget(self);
        }
    }

    // Creates an immutable borrow
    fn borrow<'a>(&'a self) -> Borrow<'a, T> {
        if self.loan.get() == MutLoan {
            fail!("Conflicting borrow of transaction variable");
        }
        
        self.loan.set(ImmLoan);
        
        Borrow::<'a,T> { 
            value: self,
            no_share: NoShare,
            no_send: NoSend,
            lifetime: ContravariantLifetime
        }
    }

    // Creates a mutable borrow
    fn borrow_mut<'a>(&'a self) -> MutBorrow<'a, T> {
        if self.loan.get() != NoLoan {
            fail!("Conflicting borrow of transaction variable");
        }
        
        self.loan.set(MutLoan);
        
        // Since the caller may modify the value, we must
        // write it back to the variable on transaction commit
        self.dirty.set(true);
        
        MutBorrow::<'a,T> { 
            value: unsafe { transmute(self) },
            no_share: NoShare,
            no_send: NoSend,
            lifetime: ContravariantLifetime
        }
    }
}

// Record in a transaction log
struct Record<T> {
    // The transaction variable
    var: Var<T>,
    // Local copy of the variable contents.
    // This is Some for nearly the entire lifetime
    // of the Record.  It's an Option so we can
    // call .take() to move out of it on commit.
    val: Option<Value<T>>,
    // Commit epoch of the variable when we read it
    read_epoch: uint,
    // Commit epoch of the variable for committing.
    // If this is the dummy value LOCKED_READ, we don't
    // have a lock at the moment.  When we successfully
    // write back to the variable, it is updated to
    // the commit epoch of the transaction.  In any case,
    // the lock is released when the record is dropped
    // by any means.
    lock_epoch: uint
}

impl<T: Send + Clone> Record<T> {
    // Creates a new transaction record.  This can fail
    // if the variable has been updated since `epoch`,
    // which is the commit epoch of the transaction
    // in progress.
    fn new(var: &Var<T>, epoch: uint) -> Option<Record<T>> {
        let epoch = var.lock_read(epoch);

        match epoch {
            Some(e) => {
                let mut record = Record {
                    var: var.clone(),
                    val: None,
                    read_epoch: e,
                    lock_epoch: e
                };
                // If something goes wrong here (e.g. `.clone()` fails),
                // drop of the record will release the lock.
                record.val = unsafe { Some(Value::new(var.read())) };
                // Now that we have the value, release our lock
                record.unlock();
                Some(record)
            }
            None => None
        }
    }
}

// Type-erased version of a VarBox<T>
enum ErasedVar {}
// Type-erased version of a Value<T>
enum ErasedValue {}

// Trait for type-erasing Record<T>
#[doc(hidden)]
trait ErasedRecord {
    // Locks the variable for commit, returning false if
    // an inconsistency was detected
    fn lock(&mut self) -> bool;
    // Writes the value back to the variable with the
    // given commit epoch if it is dirty
    fn write(&mut self, epoch: uint);
    // Unlocks the variable for commit if it was locked
    fn unlock(&mut self);
    // Verifies that the variable has not changed since it was
    // read, returning true if the transaction is still consistent
    fn verify(&self) -> bool;
    // Gets the inner Value<T> in type-erased form
    fn get(&self) -> *ErasedValue;
}

impl<T: Send + Clone> ErasedRecord for Record<T> {
    fn lock(&mut self) -> bool {
        self.lock_epoch = match self.var.lock_commit(self.read_epoch) {
            None => return false,
            Some(e) => e
        };

        true
    }

    fn write(&mut self, epoch: uint) {
        assert!(self.lock_epoch != LOCKED_READ);

        let val = self.val.take().unwrap();

        if val.is_dirty() {
            val.writeback(&self.var);
            self.lock_epoch = epoch;
        }
    }

    fn unlock(&mut self) {
        if self.lock_epoch != LOCKED_READ {
            self.var.unlock(self.lock_epoch);
            self.lock_epoch = LOCKED_READ;
        }
    }

    fn verify(&self) -> bool {
        self.var.verify(self.read_epoch)
    }

    fn get(&self) -> *ErasedValue {
        unsafe {
            transmute(self.val.as_ref().unwrap())
        }
    }
}

#[unsafe_destructor]
impl<T: Send + Clone> Drop for Record<T> {
    fn drop(&mut self) {
        // No matter how a record gets dropped, we
        // release any commit lock on the transaction variable.
        self.unlock();
    }
}

// Type used when unwinding the stack to restart a transaction
// in an inconsistent state
struct Inconsistent;

/// Transaction context
///
/// Represents a transaction in progress on the current task.
/// The only way to acquire a reference to a transaction is with
/// the `atomically` function.
pub struct Transaction {
    // The transaction log.  It maps each variable read/modified
    // by the transaction so far to a record structure which maintains
    // a task-local copy.  The keys and values are type-erased since
    // the log is a heterogenous collection.  We put it in an unsafe
    // box since it's possible to obtain a mutable reference to it
    // from an &Transaction.
    log: Unsafe<HashMap<*mut ErasedVar, ~ErasedRecord>>,
    epoch: uint,
    no_send: NoSend,
    no_share: NoShare
}

impl Transaction {
    // Gets the next global change epoch
    fn next_epoch() -> uint {
        let mut epoch;

        loop {
            epoch = unsafe { atomic_xadd_relaxed(&mut GLOBAL_EPOCH, 1) };
            if epoch != LOCKED_READ && epoch != LOCKED_COMMIT {
                break;
            }
        }

        epoch
    }

    fn new() -> Transaction {
        Transaction {
            log: Unsafe::new(HashMap::new()),
            epoch: Transaction::next_epoch(),
            no_send: NoSend,
            no_share: NoShare
        }
    }

    // Grabs a mutable reference to the transaction log from
    // an immutable self.  We are careful not to create aliases.
    unsafe fn log<'a>(&'a self) -> &'a mut HashMap<*mut ErasedVar, ~ErasedRecord> {
        &mut *self.log.get()
    }

    // Commits the log, returning true on success
    fn commit(self) -> bool {
        let log = unsafe { self.log() };

        for (_,record) in log.mut_iter() {
            if !record.lock() {
                return false;
            }
        }
        
        for (_,record) in log.mut_iter() {
            record.write(self.epoch);
        }

        // Unlocking will happen due to self being dropped
        true
    }

    // Loads a variable from the transaction log, adding it to the log first
    // if necessary.  If we detect an inconsistency (a variable updated
    // after the transaction began), we abort the transaction.
    fn load<'a, T: Send + Clone>(&'a self, var: &Var<T>) -> *mut Value<T> {
        let log = unsafe { self.log() };
        let it = var.it as *mut ErasedVar;
        let erased = log.find_or_insert_with(it, |_| {
            let record = Record::new(var, self.epoch);
            match record {
                Some(r) => ~r as ~ErasedRecord,
                None => self.abort()
            }
        });
        unsafe { transmute(erased.get()) }
    }

    /// Borrow transaction variable
    ///
    /// Borrows an immutable reference to the value of a transaction variable.
    /// 
    /// # Failure
    /// 
    /// Fails if the variable is already borrowed mutably by the current task
    pub fn borrow<'a, T: Send + Clone>(&'a self, var: &Var<T>) -> Borrow<'a, T> {
        unsafe {
            (*self.load(var)).borrow()
        }
    }

    /// Mutably borrow transaction variable
    ///
    /// Borrows a mutable reference to the value of a transaction variable.
    ///
    /// # Failure
    ///
    /// Fails if the variable is already borrowed mutably or immutably by
    /// the current task.
    pub fn borrow_mut<'a, T: Send + Clone>(&'a self, var: &Var<T>) -> MutBorrow<'a, T> {
        unsafe {
            (*self.load(var)).borrow_mut()
        }
    }

    /// Read variable
    ///
    /// Returns a copy of the value of a transaction variable.
    ///
    /// # Failure
    ///
    /// Fails if the variable is borrowed mutably by the current task.
    pub fn get<T: Send + Clone>(&self, var: &Var<T>) -> T {
        self.borrow(var).clone()
    }

    /// Write variable
    ///
    /// Sets the value of a transaction variable.
    ///
    /// # Failure
    ///
    /// Fails if the variable is borrowed by the current task.
    pub fn set<T: Send + Clone>(&self, var: &Var<T>, val: T) {
        *self.borrow_mut(var) = val;
    }

    /// Abort transaction
    ///
    /// Immediately aborts and restarts the current transaction.
    pub fn abort(&self) -> ! {
        begin_unwind_quiet(Inconsistent, file!(), line!())
    }

    fn is_consistent(&self) -> bool {
        let log = unsafe { self.log() };
        for record in log.values() {
            if !(record.verify()) {
                return false;
            }
        }

        true
    }

    /// Verify transaction consistency
    ///
    /// Verifies that the current transaction is still consistent
    /// -- that is, that it can still commit successfully.  If
    /// it is not, the transaction is immediately aborted and
    /// restarted.
    pub fn verify(&self) {
        if !self.is_consistent() {
            self.abort();
        }
    }
}

/// Perform transaction
///
/// Performs the transaction represented by the provided closure.
/// If a transaction is already in progress on the current task,
/// it is executed as a sub-transaction.
///
/// The closure is passed a handle to the transaction which can be
/// used to control it and borrow variables.
pub fn atomically<R>(f: |&Transaction| -> R) -> R {
    local_data::get(cur_transaction, |opt| match opt {
        Some(&trans) => {
            // We are already in a transaction.  Just hand
            // the current transaction handle to the closure.
            f(unsafe { transmute(trans) })
        },
        None => {
            // We need to begin a new transaction.
            let mut res;
            // Keep trying until we successfully commit
            // or fail out entirely
            loop {
                let trans = Transaction::new();
                // Run the closure with the transaction handle registered
                // in local storage
                local_data::set(cur_transaction, unsafe { transmute(&trans) });
                // Catch any exceptions.  In particular, we need to catch
                // Inconsistent so we can restart the transaction.
                let f_res = try_catch(|| f(&trans));
                local_data::pop(cur_transaction);
                match f_res {
                    Ok(r) => {
                        // Transaction succeeded so far.  Save the result
                        // of the closure to return.
                        res = r;
                        // If we succesfully commit it, we're out of here!
                        if trans.commit() {
                            break;
                        }
                        // D'oh
                        debug!("Transaction failed to commit; retrying");
                        // Drop the old result now in case it's keeping
                        // a lot of memory allocated
                        drop(res);
                    },
                    Err(e) => {
                        if e.is::<Inconsistent>() {
                            // Always retry on an inconsistency
                            debug!("Transaction aborted due to inconsistency; retrying");
                            continue;
                        } else if !trans.is_consistent() {
                            // On other failures, retry if the transaction is inconsistent,
                            // as having an inconsistent view of the transaction variables
                            // may have been the cause of the failure.
                            debug!("Transaction failed but was inconsistent; retrying");
                            continue;
                        } else {
                            // We couldn't have failed due to program observing an
                            // inconsistent state.  Re-throw.
                            // FIXME: this doesn't exactly re-throw correctly
                            // because e is already boxed.  Probably need to
                            // add another unwind function to the runtime...
                            fail!(e);
                        }
                    }
                }
            }
            res
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::TaskBuilder;

    #[test]
    fn multitask() {
        let var_a = Var::new(0);
        let var_b = Var::new(0);
        let increments = 10;
        let tasks = 10;
        let mut results = Vec::new();

        for num in range(0, tasks) {
            let a = var_a.clone();
            let b = var_b.clone();
            let mut builder = TaskBuilder::new();

            results.push(builder.future_result());

            builder.spawn(proc() {
                for _ in range(0, increments) {
                    // Atomically increment both a and b
                    let (x,y) = atomically(|trans| {
                        let new_a = a.get() + 1;
                        let new_b = b.get() + 1;
                        
                        // Verify we are in a consistent state before bothering
                        // to proceed.  This doesn't change behavior since
                        // the transaction would just fail to commit and be
                        // restarted anyway, but it allows avoiding unnecessary
                        // work.
                        trans.verify();

                        a.set(new_a);
                        b.set(new_b);
                        (new_a, new_b)
                    });
                    // x and y will always be equal because `atomically` guarantees it
                    assert_eq!(x, y);
                    debug!("{}: {} {}", num, x, y);
                }
            });
        }

        for receiver in results.move_iter() {
            let result = receiver.recv();
            assert!(result.is_ok());
        }

        assert_eq!(var_a.get(), tasks * increments);
        assert_eq!(var_b.get(), tasks * increments);
    }

    // An STM vector of STM elements
    //
    // This structure probably isn't worth exposing, but it makes
    // for a good test
    struct StimVec<T> {
        vec: Var<Vec<Var<T>>>
    }

    impl<T: Send + Clone> Clone for StimVec<T> {
        fn clone(&self) -> StimVec<T> {
            StimVec {
                vec: self.vec.clone()
            }
        }
    }

    impl<T: Send + Clone> StimVec<T> {
        pub fn new() -> StimVec<T> {
            StimVec {
                vec: Var::new(Vec::new())
            }
        }

        pub fn get(&self, i: uint) -> T {
            atomically(|trans| {
                let v = trans.borrow(&self.vec);
                trans.get(v.get(i))
            })
        }

        pub fn set(&self, i: uint, val: T) {
            atomically(|trans| {
                let v = trans.borrow(&self.vec);
                trans.set(v.get(i), val.clone());
            });
        }

        pub fn push(&self, val: T) {
            atomically(|trans| {
                let mut v = trans.borrow_mut(&self.vec);
                v.push(Var::new(val.clone()))
            });
        }

        pub fn pop(&self) -> Option<T> {
            atomically(|trans| {
                let mut v = trans.borrow_mut(&self.vec);
                match v.pop() {
                    Some(ref e) => Some(trans.get(e)),
                    None => None
                }
            })
        }

        pub fn mutate_all(&self, f: |&T| -> T) {
            atomically(|trans| {
                let v = trans.borrow(&self.vec);
                for e in v.iter() {
                    // This is a little uglier than it
                    // should be because the deref traits
                    // are kind of awkward.
                    let mut val = trans.borrow_mut(e);
                    *val = f(val.deref());
                }
            });
        }
    }

    #[test]
    fn stimvec() {
        let v1: StimVec<int> = StimVec::new();
        let v2: StimVec<int> = StimVec::new();
        let mut results = Vec::new();
        let tasks = 10;

        // Initialize the vector in one transaction
        atomically(|_| {
            // Push some elements
            v1.push(1);
            v1.push(2);
            v1.push(3);

            // Change some elements around
            v1.mutate_all(|&x| x*2);
            v1.set(1, -v1.get(1));
        });

        for _ in range(0, tasks) {
            let mut builder = TaskBuilder::new();
            results.push(builder.future_result());

            let my_v1 = v1.clone();
            let my_v2 = v2.clone();

            builder.spawn(proc() {
                // Atomically pop all elements off one vec
                // and push them on the other
                let count = atomically(|_| {
                    let mut pop_count: uint = 0;

                    loop {
                        match my_v1.pop() {
                            Some(e) => {
                                debug!("Popped {}", e);
                                pop_count += 1;
                                my_v2.push(e);
                            },
                            None => break
                        }
                    }

                    pop_count
                });

                // Since the above is atomic, either we popped
                // them all or we didn't pop any
                assert!(count == 3 || count == 0);
            });
        }

        for receiver in results.move_iter() {
            let result = receiver.recv();
            assert!(result.is_ok());
        }

        assert_eq!(v2.get(0), 6);
        assert_eq!(v2.get(1), -4);
        assert_eq!(v2.get(2), 2);
    }
}
