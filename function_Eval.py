from objfun import objfun

def feval(fname,soln,Inp,Tar,test_dat,test_tar):
    if fname == 'objfun':
        out = objfun(soln,Inp,Tar,test_dat,test_tar)
    else:
        out = None
    return out

