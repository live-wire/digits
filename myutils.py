from sklearn.externals import joblib


def backup_util(backupname,func,*args):
    try:
        ret = joblib.load(backupname)
    except:
        ret = func(*args)
        joblib.dump(ret,backupname)
    return ret

