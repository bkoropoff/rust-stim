RLIB=$(shell rustc --crate-type=rlib --crate-file-name lib.rs)
DYLIB=$(shell rustc --crate-type=dylib --crate-file-name lib.rs)
RUSTFLAGS=-O -g

all: .libs doc

$(RLIB) $(DYLIB): .libs
	@touch $@

.libs: lib.rs
	rustc $(RUSTFLAGS) --crate-type=rlib,dylib lib.rs
	@touch $@

test-stim: lib.rs
	rustc $(RUSTFLAGS) --test -o $@ $<

test: test-stim
	./test-stim

doc: lib.rs
	@rm -rf $@
	rustdoc -o $@ $<
	@touch $@

clean:
	rm -f *.so *.a *.rlib .libs test-stim
	rm -rf doc

.PHONY: clean all test
