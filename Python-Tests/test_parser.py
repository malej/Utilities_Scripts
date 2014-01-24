import argparse

parser = argparse.ArgumentParser()

#parser.add_argument('echo',help='prints the string you use here')
#parser.add_argument('square',help='square the argument')
#parser.add_argument('square', type=float, help="display a square of a floating point number")
#parser.add_argument('-v','--verbose', help='increase output verbosity', action="store_true")
parser.add_argument('base', type=float, help='the base')
parser.add_argument('exp', type=float, help='the exponent')
parser.add_argument('-v','--verbose', type=int, choices=[0,1,2], 
                    default=0, help="increase verbosity type from 0 to 2")
args = parser.parse_args()

answer = args.base**args.exp

if args.verbose == 0:
    print "verbosity is OFF, since its value is  ", args.verbose

elif args.verbose == 1:
    print "verbosity is ON at level ", args.verbose, " ... answer is: ", answer

elif args.verbose ==2: 
    print  "verbose is ON at level {}, so {}^{}={}".format(args.verbose, args.base,\
                                                               args.exp, answer)
