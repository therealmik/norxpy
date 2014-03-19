norxpy
======

Python/Numpy implementation of https://norx.io/

This is intended as a reference/educational implementation of
NORX.

To create an instance of the round function, you can call

>> F = norx.NORX_F(32)
or
>> F = norx.NORX_F(64)

The NORX_F class has no real state - just initialized with some
variables (rotation offsets, data types, etc).

You can get a new state array by calling:

>> state = F.new(nonce, key)

And just call:

>> F(state)

to perform a single round.

I've included a handy ufunc 'bitcount' which will count how many
bits are set.

