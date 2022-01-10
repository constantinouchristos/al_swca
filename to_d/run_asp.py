import argparse
import pathlib
import os



def try_this(a=None,
             b=None,
             c=None,
             d=None):
    
    s = 0 
    
    for i in range(b):
        
        s += a
        
    if d:
        
        s += c
        
    return s




if  __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-al_it",
                        "--active_learning_iterations",
                        help="parameter a",
                        type=float,
                        default=0.5,
                       )
    
    parser.add_argument("-b",
                        help="parameter b",
                        choices=[1, 2],
                        default=1,
                        type=int,
                       )
    
    parser.add_argument("-c",
                        help="parameter c",
                        type=float,
                        default=0.5,
                       )
    
    parser.add_argument("-d",
                        "--d",
                        help="parameter d",
                        action="store_true"
                       )
    
#     parser.add_argument("-path",
#                         "--p",
#                         help="parameter p",
#                         type=pathlib.Path,
#                         required=True
#                        )
    
    parser.add_argument("-gg",
                        help="parameter b",
                        choices=['apple', 'banana'],
                        default=1,
                        type=str,
                       )
    
    args = parser.parse_args()
    
    result = try_this(args.active_learning_iterations,args.b,args.c,args.d)
    print(result)
#     print(args.p)
    print(args.d)
#     print(type(args.p))
    print(args)
    print(args.__dict__)
    
#     print(os.listdir(args.p))
    
    