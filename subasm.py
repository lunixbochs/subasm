import re

## var
# x
# x[0]
# x[x]

## immediates
# 0
# 0x1
# -1
# 'a'
# [0, 1, 2, 3] # global array of 32-bit imms
# [aaaaaaa]    # global array of 32-bit chars

## operators
# + - * / %

## labels
# label:

## jumps
# jmp label
# jmp label if var <conditional> var_or_imm

## conditionals
# == != < > <= >=

# var parsing
var_embed = r'\w+(\[[^\]]+\])?'
imm_embed = r"(-?(0x[0-9a-fA-F]+|[0-9]+|'[a-z]')|len\(\w+\))"
var_imm_embed = r'({}|{})'.format(var_embed, imm_embed)

array_embed = r"\[([a-z]+|{}(,\s*{})*)\]".format(imm_embed, imm_embed)

var_re = re.compile(r'^((?P<name>\w+?)(?P<index>\[[^\]]+\])?)$')
imm_re = re.compile(r"^((?P<numeric>-?(0x[0-9a-fA-F]+|[0-9]+))|'(?P<char>[a-z])')$")
array_re = re.compile(r"\[((?P<chars>[a-z]+)|(?P<elements>[^]]+))\]")

ops = '-=+*/%'
op_embed = '(' + ('|'.join([re.escape(o) for o in ops])) + ')'
conds = ['==', '!=', '<', '<=', '>=']
cond_embed = '(' + ('|'.join([re.escape(o) for o in conds])) + ')'

label_re = re.compile(r'^(?P<name>\w+):$')
assign_line_re = re.compile(r'^(?P<dst>{})\s*=\s*(?P<src>{}|{})$'.format(var_embed, var_imm_embed, array_embed))
expr_line_re = re.compile(r'^(?P<dst>{})\s*=\s*(?P<src_a>{})\s*(?P<op>{})\s*(?P<src_b>{})$'.format(var_embed, var_imm_embed, op_embed, var_imm_embed))
jump_line_re = re.compile(r'^jmp\s*(?P<label>\w+)(\s*if\s*(?P<var>{})\s*(?P<cond>{})\s*(?P<cmpto>{}))?$'.format(var_embed, cond_embed, var_imm_embed))

def test_regexes():
    for test in ['x', 'x[0]', 'x[x]']:
        match = var_re.match(test)
        print(test, match.groups())

    lines = r'''
    a = 1
    a = a * 1
    a[0] = 2
    a = a[1]
    1:
    qqqqqq:
    jmp 1
    jmp 1 if a == 2
    '''.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        print(line)
        match = label_re.match(line)
        if match: print('label', match.groupdict())
        match = assign_line_re.match(line)
        if match: print('assign', match.groupdict())
        match = expr_line_re.match(line)
        if match: print('expr', match.groupdict())
        match = jump_line_re.match(line)
        if match: print('jump', match.groupdict())
        print()

class Assign:
    def __init__(self, lineno, dst, src):
        self.lineno = lineno
        self.dst = dst
        self.src = src

    def __repr__(self):
        return "{} = {}".format(self.dst, self.src)

class Expr:
    def __init__(self, lineno, dst, src_a, op, src_b):
        self.lineno = lineno
        self.dst = dst
        self.src_a = src_a
        self.op = op
        self.src_b = src_b

    def __repr__(self):
        return "{} = {} {} {}".format(self.dst, self.src_a, self.op, self.src_b)

class Jump:
    def __init__(self, lineno, label, var=None, cond=None, cmpto=None):
        self.lineno = lineno
        self.label = label
        self.var = var
        self.cond = cond
        self.cmpto = cmpto

    def __repr__(self):
        if self.var and self.cond and self.cmpto:
            return "jmp {} if {} {} {}".format(self.label, self.var, self.cond, self.cmpto)
        else:
            return "jmp {}".format(self.label)

class Label:
    def __init__(self, lineno, name):
        self.lineno = lineno
        self.name = name

    def __repr__(self):
        return "{}:".format(self.name)

def parse(source):
    lines = source.split('\n')
    parsed = []
    for lineno, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        massign = assign_line_re.match(line)
        mexpr   = expr_line_re.match(line)
        mjump   = jump_line_re.match(line)
        mlabel  = label_re.match(line)
        if massign:
            parsed.append(Assign(lineno, **massign.groupdict()))
        elif mexpr:
            parsed.append(Expr(lineno, **mexpr.groupdict()))
        elif mjump:
            parsed.append(Jump(lineno, **mjump.groupdict()))
        elif mlabel:
            parsed.append(Label(lineno, **mlabel.groupdict()))
        else:
            raise Exception("Couldn't parse line {}: {}".format(lineno, line))
    return parsed

class Var:
    def __init__(self, name=None, index=None):
        self.name = name
        self.index = index

    def __repr__(self):
        name = self.name
        if self.index:
            name = name + self.index
        return name

class Immediate:
    def __init__(self, numeric=0, char=None):
        value = numeric
        if char is not None:
            value = ord(char)
        else:
            value = int(value, 0)
        self.value = value

    def __repr__(self):
        return str(self.value)

class Array:
    def __init__(self, chars=None, elements=()):
        if chars is not None:
            elements = [ord(c) for c in chars]
        self.values = elements

    def __repr__(self):
        return repr(list(self.values))

def get_var(s):
    mvar = var_re.match(s)
    if mvar:
        return Var(**mvar.groupdict())
    else:
        raise Exception("failed to parse var: {}".format(s))

def get_var_imm(s):
    mvar = var_re.match(s)
    mimm = imm_re.match(s)
    marray = array_re.match(s)
    if mimm: return Immediate(**mimm.groupdict())
    elif marray: return Array(**marray.groupdict())
    elif mvar: return Var(**mvar.groupdict())
    else:
        raise Exception("failed to parse var or imm: {}".format(s))

class Program:
    def __init__(self, code_base=0x30200000, data_base=0x30400000):
        self.code_base = code_base
        self.data_base = data_base
        self.pc = code_base
        self.bump_alloc = data_base

        self.code = []
        self.current_element = None
        self.label_backpatch = []
        self.labels = {}
        self.memory = []
        self.sourcemap = {}
        self.variables = {}
        self.constants = {}

    def ins(self, a, b, c=None):
        self.sourcemap[self.pc] = self.current_element
        if c is None:
            c = self.pc + 12
        self.code += [a, b, c]
        self.pc += 12

    def alloc(self, value=0):
        ret = self.bump_alloc
        self.bump_alloc += 4
        self.memory += [value]
        return ret

    def calloc(self):
        tmp = self.alloc()
        self.ins(tmp, tmp)
        return tmp

    def var_addr(self, name):
        if not name in self.variables:
            self.variables[name] = self.alloc()
        return self.variables[name]

    def imm_addr(self, imm):
        if isinstance(imm, Immediate):
            imm = imm.value

        if isinstance(imm, int):
            if not imm in self.constants:
                tmp = self.alloc(value=imm)
                self.constants[imm] = tmp
                return tmp
            return self.constants[imm]
        elif isinstance(imm, Array):
            addr = self.alloc(value=self.bump_alloc + 4)
            for n in imm.values:
                self.alloc(value=n)
            return addr
        elif isinstance(imm, Var):
            addr = self.var_addr(imm.name)
            if imm.index:
                # when reading an index (pointer), JIT a copy operation into a temporary variable
                return self.copy_index(addr, imm.index)
            return addr
        else:
            raise Exception("invalid immediate type: {}".format(imm))

    def copy(self, src):
        # tmp = -src; dst = -tmp
        tmp = self.calloc()
        dst = self.calloc()
        self.ins(src, tmp)
        self.ins(tmp, dst)
        return dst

    def assign(self, dst, src):
        if src == self.constants.get(0):
            self.ins(dst, dst)
            return
        # tmp = -src; dst = 0; dst -= src
        tmp = self.calloc()
        self.ins(src, tmp)
        self.ins(dst, dst)
        self.ins(tmp, dst)

    def get_index_addr(self, addr, index):
        index = index.strip('[]')
        if index.lstrip('-').isdigit():
            index = int(index) * 4
            index_addr = self.imm_addr(index)
        else:
            index = str(index)
            if not index in self.variables:
                raise Exception('index variable [{}] not defined before use'.format(index))
            index_var_addr = self.variables[index]
            four = self.imm_addr(4)
            index_addr = self.calloc()
            self.mul(index_addr, index_var_addr, four)
        tmp = self.calloc()
        self.add(tmp, addr, index_addr)
        return tmp

    def copy_index(self, src, index):
        neg_dst = self.calloc()
        index_addr = self.get_index_addr(src, index)
        neg_src = self.calloc()
        self.ins(index_addr, neg_src)

        # need self-modifying code to read index...
        jit_slot2 = self.pc + 12 * 2

        self.ins(jit_slot2, jit_slot2)
        self.ins(neg_src, jit_slot2)
        self.ins(0, neg_dst)

        dst = self.calloc()
        self.ins(neg_dst, dst)
        return dst

    def assign_index(self, dst, src, index):
        index_addr = self.get_index_addr(dst, index)

        neg_src = self.calloc()
        self.ins(src, neg_src)
        neg_dst = self.calloc()
        self.ins(index_addr, neg_dst)
        # print('neg_src', hex(neg_src), 'neg_dst', hex(neg_dst))

        # need self-modifying code to write index...
        dst1_slot1 = self.pc + 12 * 6
        dst1_slot2 = dst1_slot1 + 4
        dst2_slot2 = self.pc + 12 * 7 + 4

        self.ins(dst1_slot1, dst1_slot1)
        self.ins(dst1_slot2, dst1_slot2)
        self.ins(dst2_slot2, dst2_slot2)
        self.ins(neg_dst, dst1_slot1)
        self.ins(neg_dst, dst1_slot2)
        self.ins(neg_dst, dst2_slot2)
        self.ins(0, 0)
        self.ins(neg_src, 0)

    def sub(self, dst, a, b):
        if a == dst:
            self.ins(b, dst)
            return
        tmp = self.copy(b)
        self.assign(dst, a)
        self.ins(tmp, dst)

    def add(self, dst, a, b):
        if a == dst:
            tmp = self.calloc()
            self.ins(b, tmp)
            self.ins(tmp, dst)
            return
        tmp1 = self.calloc()
        tmp2 = self.calloc()
        self.ins(a, tmp1)
        self.ins(b, tmp2)
        self.ins(dst, dst)
        self.ins(tmp1, dst)
        self.ins(tmp2, dst)

    def mul(self, dst, a, b):
        i = self.calloc()
        acc = self.calloc()
        neg_one = self.imm_addr(-1)
        # loop B times, adding A each time
        # i = -b - -1
        # while i < 0
        #   acc -= a
        # i -= -1
        # dst = -acc
        self.ins(b, i)
        self.ins(neg_one, i)
        loop = self.pc
        self.ins(a, acc)
        self.ins(neg_one, i, loop)
        # dst = 0; dst -= acc
        self.ins(dst, dst)
        self.ins(acc, dst)

    def div(self, dst, a, b):
        # loop, subtracting B from A and incrementing ACC each time, until A < 0
        A = self.calloc()
        B = self.calloc()
        ACC = self.calloc()
        one = self.imm_addr(1)
        neg_one = self.imm_addr(-1)
        # set acc to 1, so the first dec sets acc to 0
        self.ins(neg_one, ACC)
        self.ins(a, A)
        self.ins(b, B)

        loop = self.pc
        self.ins(one, ACC)
        self.ins(B, A, loop)
        # dst = 0; dest -= ACC
        self.ins(dst, dst)
        self.ins(ACC, dst)

    def mod(self, dst, a, b):
        # loop, subtracting B from A, until A < 0
        # then add B to A and the result is the modulo
        A = self.calloc()
        B = self.calloc()
        self.ins(a, A)
        self.ins(b, B)
        loop = self.pc
        self.ins(B, A, loop)
        self.ins(b, A)
        # dst = 0; dest -= ACC
        self.ins(dst, dst)
        self.ins(A, dst)

    def assemble(self, code):
        neg1_addr = self.imm_addr(-1)
        zero_addr = self.imm_addr(0)
        elements = parse(code)
        for el in elements:
            self.current_element = el
            if isinstance(el, Assign):
                dst_var = get_var(el.dst)
                dst = self.var_addr(dst_var.name)
                src = self.imm_addr(get_var_imm(el.src))
                if dst_var.index:
                    self.assign_index(dst, src, dst_var.index)
                else:
                    self.assign(dst, src)
            elif isinstance(el, Expr):
                # + - * / %
                dst_var = get_var(el.dst)
                src_a = self.imm_addr(get_var_imm(el.src_a))
                src_b = self.imm_addr(get_var_imm(el.src_b))

                # if we're assigning to an index (pointer)
                # for simplicity, store to a temporary first then use assign_index()
                if dst_var.index:
                    dst = self.calloc()
                else:
                    dst = self.imm_addr(dst_var)

                op = el.op
                if op == '-':
                    self.sub(dst, src_a, src_b)
                elif op == '+':
                    self.add(dst, src_a, src_b)
                elif op == '*':
                    self.mul(dst, src_a, src_b)
                elif op == '/':
                    self.div(dst, src_a, src_b)
                elif op == '%':
                    self.mod(dst, src_a, src_b)
                else:
                    raise Exception("unknown op: {}".format(op))

                if dst_var.index:
                    self.assign_index(self.var_addr(dst_var.name), dst)
            elif isinstance(el, Jump):
                if el.cond and el.cmpto:
                    var = self.imm_addr(get_var(el.var))
                    cmpto = self.imm_addr(get_var_imm(el.cmpto))
                    cond = el.cond
                    if cond == '==':
                        # TODO
                        # self.sub(tmp, var, cmpto)
                        # self.sub(tmp2, cmpto, var)
                        raise Exception("Unsupported jump cond type: {}".format(cond))
                    elif cond == '!=':
                        # TODO
                        raise Exception("Unsupported jump cond type: {}".format(cond))
                    elif cond == '<':
                        tmp = self.calloc()
                        self.sub(tmp, var, cmpto)
                        neg_one = self.imm_addr(-1)
                        self.ins(neg_one, tmp, 0)
                    elif cond == '>':
                        tmp = self.calloc()
                        self.sub(tmp, cmpto, var)
                        neg_one = self.imm_addr(-1)
                        self.ins(neg_one, tmp, 0)
                    elif cond == '<=':
                        tmp = self.copy(var)
                        self.ins(cmpto, tmp, 0)
                    elif cond == '>=':
                        tmp = self.copy(cmpto)
                        self.ins(var, tmp, 0)
                    else:
                        raise Exception("Unknown jump cond type: {}".format(cond))
                else:
                    # unconditional jump
                    self.ins(zero_addr, neg1_addr, 0)

                backpatch = self.pc - 4
                self.label_backpatch.append((el.label, backpatch))
            elif isinstance(el, Label):
                self.labels[el.name] = self.pc
            else:
                raise Exception("Unknown element: {}".format(el))

        for label, pc in self.label_backpatch:
            self.code[(pc - self.code_base) // 4] = self.labels[label]

        # print([hex(n) for n in self.code])
        # print([hex(n) for n in self.memory])
        # print({k: hex(v) for k, v in self.variables.items()})

def sim(prog):
    mem = prog.memory
    code = prog.code
    sourcemap = prog.sourcemap
    variables = prog.variables
    rvars = {v: k for k, v in variables.items()}

    CODE_ADDR = 0x30200000
    MEM_ADDR  = 0x30400000
    HEAP_ADDR = 0x40000000

    heap = [0] * 1024

    def read(addr):
        # print('read {:x}'.format(addr))
        if CODE_ADDR <= addr < CODE_ADDR + len(code) * 4:
            return code[(addr - CODE_ADDR) // 4]
        elif MEM_ADDR <= addr < MEM_ADDR + len(mem) * 4:
            return mem[(addr - MEM_ADDR) // 4]
        elif HEAP_ADDR <= addr < HEAP_ADDR + len(heap) * 4:
            return heap[(addr - HEAP_ADDR) // 4]
        else:
            raise Exception('segfault read at {:#x}'.format(addr))

    def write(addr, value):
        # print('write {:x} = {:x}'.format(addr, value))
        if CODE_ADDR <= addr < CODE_ADDR + len(code) * 4:
            code[(addr - CODE_ADDR) // 4] = value
        elif MEM_ADDR <= addr < MEM_ADDR + len(mem) * 4:
            mem[(addr - MEM_ADDR) // 4] = value
        elif HEAP_ADDR <= addr < HEAP_ADDR + len(heap) * 4:
            heap[(addr - HEAP_ADDR) // 4] = value
        else:
            raise Exception('segfault write at {:#x}'.format(addr))

    ip = CODE_ADDR
    last_element = None
    try:
        while True:
            element = sourcemap.get(ip, None)
            if element and element != last_element:
                print('[+] {}'.format(element))
                last_element = element
            next_ip = ip + 12
            try:
                a = read(ip)
                b = read(ip + 4)
                c = read(ip + 8)
            except Exception:
                break
            print('  ip = {:x}, a={:x}, b={:x}, c={:x}'.format(ip, a, b, c), end='')

            val = read(b) - read(a)
            write(b, val)

            if b in rvars:
                print(' | {} = {:#x}'.format(rvars[b], val))
            else:
                print(' | {:#x}'.format(val))
            if val <= 0 and c != next_ip:
                print('  jmp {:x}'.format(c))
                next_ip = c
            ip = next_ip
            # print('mem:  ' + ' '.join([hex(n) for n in mem]))
            # print('heap: ' +' '.join([hex(n) for n in heap[:8]]))
    except Exception as e:
        print()
        print(e)
    print()
    print('heap: ' +' '.join([hex(n) for n in heap[:8]]))
    print('variables:')
    for name, addr in variables.items():
        print('  {} = {:x}'.format(name, read(addr)))

if __name__ == '__main__':
    prog1 = '''
    HEAP = 0x40000000
    a = 1
    b = a + 1
    HEAP[0] = a
    '''

    prog2 = '''
    a = 2
    b = 2
    c = a * b
    '''
    
    prog3 = '''
    moves = ['n', 'n', 'a', 's', 'l']
    moves = [nnasl]
    tick = HEAP[2]
    move = tick % 5
    HEAP[0] = moves[move]
    jmp label
    '''

    prog4 = '''
    HEAP = 0x40000000
    a = 5
    b = 0
    label:
        a = a - 1
        b = b + 1
    jmp label if a >= 0

    HEAP[0] = b
    '''

    prog5 = '''
    HEAP = 0x40000000
    HEAP[2] = 1

    tick = HEAP[2]
    tick_mod = tick % 2

    jmp move1 if tick_mod < 1
    jmp move2

    move1:
    HEAP[0] = 'n'
    jmp exit

    move2:
    HEAP[0] = 'l'
    jmp exit

    exit:
    ZERO = 0
    ZERO[0] = 0
    '''

    prog6 = '''
    HEAP = 0x40000000
    HEAP[2] = 0

    moves = [abcdef]
    tick = HEAP[2]

loop:
    tick = tick + 1
    move = tick % 2

    c = moves[move]
    HEAP[0] = c
    jmp loop if tick < 6

exit:
    ZERO = 0
    ZERO[0] = 0
    '''

    prog7 = '''
    HEAP = 0x40000000
    HEAP[2] = 1

    moves = [lrla]
    tick = HEAP[2] % 4

    move = moves[tick]
    HEAP[0] = move
    '''

    prog = prog6
    print('Program:')
    print(prog)
    print()
    p = Program()
    p.assemble(prog)
    sim(p)
