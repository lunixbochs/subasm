a = 1
a = 0
a = 'a'
a = 0x00000000
a = 1
a = -1
asdfhjklasdf = 1

a = HEAP
a[0] = 'n'

moves = [nnasl]
tick = HEAP[2]
tick = HEAP + 8
tick = tick[0]

move = tick % len(array)
HEAP[0] = moves[move]

label:
    HEAP = 0
    STACK = 0
    
    a = 1 
    a = a + 1
    a = a * 4 
    jmp label 
    jmp label if a == 1
    
    if a != 1
    if a == 1
    if a < b
    if a <= b
    if a > b
    if a >= b
