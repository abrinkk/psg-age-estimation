

def test_fun(a):
    a['a'] = a['a'] + 1.5
    return a

a = {'a': 1.0, 'b': 2.0}

print(a)

q = a

z = test_fun(q)

print(a,q,z)