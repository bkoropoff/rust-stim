rust-stim
=========

Prototype STM library for Rust.

**Note: The Rust devs seem generally opposed to exposing
 unwinding to library and application code (and not without
 good reason), so I will probably have to abandon this approach.**

This is incomplete and experimental.  It unfortunately
requires a hacked-up version of libstd so that I can
recover from unwinds:

https://github.com/bkoropoff/rust/tree/unwind-hacks

API Docs
---------

http://bkoropoff.github.io/rust-stim/doc/stim/index.html

