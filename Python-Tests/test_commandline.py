import os

compiler = 'clang '
linker = compiler
cFlags = '-c -O3 '
lFlags = '-o '

execute = './'

filename = 'test_commandline'

source = '.c '
object = '.o '
binary = '.out '

os.system(compiler + cFlags + filename + source)
os.system(linker + lFlags + filename + binary + filename + object)

print 'DONE compiling and linking'

print 'NOW lets run it: ...'
os.system(execute + filename + binary)              
