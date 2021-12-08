"""
Set of utilities for Gaussian basis sets,
initially written by Maxim Losev, 2010,
supported by Evgeny Blokhin, 2011-2013
FIXME
ugly variable names
"""
import logging
import copy
import math
from pprint import pformat

from pyparsing import (
    LineStart, LineEnd, Regex, alphas, alphanums, Word, Keyword, CaselessKeyword,
    Literal, Group, SkipTo, OneOrMore, NotAny, Optional, pyparsing_common as ppc
)
from ase.data import chemical_symbols, atomic_numbers
import numpy as np


logging.basicConfig(level=logging.INFO)

def convertnum(s, l, toks):
    # debug('number_toks:%s',toks)
    n = toks[0]
    try:
        return int(n)
    except ValueError:
        return float(n.replace('d', 'e').replace('D', 'E'))


# Basic tags

eos = LineStart().suppress()
eol = LineEnd().suppress()
eol.setDebug(False)

Number = Regex('[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eEdD][-+]\d+)?') # number
Number.setParseAction(convertnum)

NaN = Keyword('NaN').setParseAction(lambda t: np.nan_to_num(np.nan))

Str = Word(alphas)
Str.setParseAction(ppc.upcaseTokens)
StrNum = Regex('[a-zA-Z]') + Word(alphanums)

# Literals

Lminus = Literal('-')
Lplus = Literal('+')
Lslash = Literal('/')
Lbslash = Literal('\\')
Leq = Literal('=')
Lden = Literal('g/cm^3')
Llpar = Literal('(')
Lrpar = Literal(')')
Lcolon = Literal(':')
Lstar = Literal('*')
Ldot = Literal('.')
Lpipe = Literal('|')

# Tags for d12

comment = Group(NotAny(eol) + Literal('!') + SkipTo(eol)).suppress() # comment
kw = Group(Regex('[A-Za-z][a-zA-Z0-9]+') + Optional(comment) + eol) # any keyword
first_line = eos + SkipTo(eol) + eol # first line with title
block_n = Group(OneOrMore(NotAny(eol) + Number) + Optional(comment)
                + eol) # sequence of number until end of line

# Basis grammar

comment = Group(Literal('!') + SkipTo(eol)).suppress()

l = CaselessKeyword('S') | CaselessKeyword('P') | CaselessKeyword('D') \
    | CaselessKeyword('F') | CaselessKeyword('G')
l_sp = CaselessKeyword('SP')

shell = Group(l.setParseAction(ppc.upcaseTokens) + Number + Number)
shell_sp = Group(l_sp.setParseAction(ppc.upcaseTokens) + Number + Number)

exponent = Group(Number + Number)
exponent_sp = Group(Number + Number + Number)

bs = Group(shell + OneOrMore(exponent))
bs_sp = Group(shell_sp + OneOrMore(exponent_sp))

endbs = Word('*').suppress()

basis_set = Group(Str.setResultsName('bdescr') + Number + OneOrMore(bs
                  | bs_sp).setResultsName('basis') + Optional(endbs))

ecp_body = OneOrMore(eol + SkipTo(eol).suppress() + Group(Number
                     + OneOrMore(Group(Number + Number + Number))))

ecp = Group(Str.setResultsName('edescr') + Number.suppress()
            + SkipTo(Number).suppress() + Number.setResultsName('lmax')
            + Number.setResultsName('core')
            + ecp_body.setResultsName('ecp'))

# define grammar here

grammar = Optional(OneOrMore(comment)) + Optional(endbs) \
    + Group(OneOrMore(basis_set)) + Optional(OneOrMore(comment)) \
    + Optional(Group(OneOrMore(ecp)))

lstr = {
    'S': 0,
    'SP': 1,
    'P': 2,
    'D': 3,
    'F': 4,
    'G': 5,
}

lecpstr = {
    'S': 0,
    'P': 1,
    'D': 2,
    'F': 3,
    'G': 4,
    'H': 5,
}


class ECP(object):

    def __init__(
        self,
        lmax,
        core,
        z,
        gtfs,
        ):
        self.lmax = lmax
        self.core = core
        self.z = z
        self.gtfs = gtfs

        # TODO: add descr attribute

    def __repr__(self):
        output = {}

        for gtf in self.gtfs:
            if gtf.l == self.lmax:
                key = str([i[0] for i in lecpstr.items() if i[1]
                          == self.lmax][0])
                if key in output:
                    output[key].append(gtf)
                else:
                    output[key] = [gtf]
            else:
                key = str([i[0] for i in lecpstr.items() if i[1]
                          == gtf.l][0]) + '-' + str([i[0] for i in
                        lecpstr.items() if i[1] == self.lmax][0])
                if key in output:
                    output[key].append(gtf)
                else:
                    output[key] = [gtf]
        return pformat(output)

    def crystal_input(self):
        output = 'INPUT\n%#.0f' % self.z
        counter = 0
        for gtf in self.gtfs:
            counter += 1
            output += ' ' + str(len(gtf.pgtfs))

        assert counter <= 6

        for i in range(6 - counter):
            output += ' 0'

        output += '\n'

        for gtf in self.gtfs:
            output += str(gtf.crystal_input()) + '\n'

        return output.rstrip()

    def gaussian_input(self):
        output = 'ECP%5d%5d\n' % (self.lmax, self.core)

        for gtf in self.gtfs:
            if gtf.l == self.lmax:
                output += '%s POTENTIAL\n' % [i[0] for i in
                        lecpstr.items() if i[1] == self.lmax][0]
            else:
                output += '%s-%s POTENTIAL\n' % ([i[0] for i in
                        lecpstr.items() if i[1] == gtf.l][0], [i[0]
                        for i in lecpstr.items() if i[1]
                        == self.lmax][0])

            output += '%d\n' % len(gtf.pgtfs)
            output += str(gtf.gaussian_input()) + '\n'

        return output


class GTF(object):

    def __init__(
        self,
        l,
        pgtfs,
        f=1.0,
        ecp=False,
        nelec=0.0,
        ):
        self.l = l
        self.pgtfs = pgtfs
        self.f = f
        self.ecp = ecp
        self.nelec = nelec

    def __repr__(self):

        if not self.ecp:
            output = [self.f, self.pgtfs]
        else:

            output = self.pgtfs

        return pformat(output)

    def crystal_input(self, nelec=0):

        if not self.ecp:
            if self.nelec:
                output = '0 ' + str(self.l) + ' ' \
                    + str(len(self.pgtfs)) + ' %#.1f %#.2f\n' \
                    % (self.nelec, self.f)
            else:
                output = '0 ' + str(self.l) + ' ' \
                    + str(len(self.pgtfs)) + ' %#.1f %#.2f\n' % (nelec,
                        self.f)
        else:

            output = ''

        for i in self.pgtfs:
            output += str(i.crystal_input()) + '\n'

        return output.rstrip()

    def gaussian_input(self):
        if not self.ecp:
            output = '%s%5d%10.2f\n' % ([i[0] for i in lstr.items()
                    if i[1] == self.l][0], len(self.pgtfs), self.f)
        else:
            output = ''

        for i in self.pgtfs:
            output += str(i.gaussian_input()) + '\n'

        return output.rstrip()


class PGTF(object):

    def __init__(
        self,
        a,
        c,
        r=None,
        lim=None,
        ):
        self.a = a

        if isinstance(c, float):
            self.c = [c]
        elif isinstance(c, list):
            self.c = c
        else:
            raise RuntimeError

        self.r = r
        self.opt = False
        self.lim = lim

    def __repr__(self):
        output = []

        if self.r or self.r == 0:
            output.append(self.r)

        output.append(self.a)
        output.extend(self.c)

        return pformat(output)

    def crystal_input(self):
        output = '%20.10f' % self.a

#        if self.opt:
#            if self.lim:
#                output='%20s'%('@<%.10f:%.5f>'%(self.a,self.lim))
#            else:
#                output='%20s'%('@<%.10f>'%(self.a))

        if len(self.c) == 2:
            output += '%20.10f%20.10f' % (self.c[0], self.c[1])
        elif len(self.c) == 1:
            output += '%20.10f' % self.c[0]
        else:
            raise RuntimeError
        if self.r or self.r == 0:
            output += '%10d' % (self.r - 2)

        return output

    def gaussian_input(self):
        output = ''

        if self.r or self.r == 0:
            output += '%10d' % self.r
        output += '%20.10f' % self.a

        if len(self.c) == 2:
            output += '%20.10f%20.10f' % (self.c[0], self.c[1])
        elif len(self.c) == 1:
            output += '%20.10f' % self.c[0]
        else:
            raise RuntimeError

        return output


class Basis(object):

    def __init__(self, gtfs):
        self.gtfs = gtfs
        self.electrons = 0
        self.history = []

        # TODO: add descr attribute

    def __repr__(self):
        output = {}

        for gtf in self.gtfs:
            key = str([i[0] for i in lstr.items() if i[1] == gtf.l][0])

            if key in output:
                output[key].append(gtf)
            else:
                output[key] = [gtf]

        return pformat(output)

    def undo(self):
        if self.history:
            self.gtfs = self.history.pop(-1)

        return self

    def add2history(self):
        if self.history:
            if self.gtfs != self.history[-1]:
                self.history.append(copy.copy(self.gtfs))
        else:
            self.history.append(copy.copy(self.gtfs))

        return self

    def mark4opt(
        self,
        a=1.0,
        values=[],
        limit=0.0,
        ):
        self.add2history()

        vs = ['%.2f' % i for i in values]

        for gtf in self.gtfs:
            if len(gtf.pgtfs) == 1:
                if values:
                    if '%.2f' % gtf.pgtfs[0].a in vs:
                        gtf.pgtfs[0].opt = True
                        if limit:
                            gtf.pgtfs[0].lim = limit
                else:
                    if gtf.pgtfs[0].a <= a:
                        gtf.pgtfs[0].opt = True
                        if limit:
                            gtf.pgtfs[0].lim = limit

        return self

    def add4opt(
        self,
        up=False,
        down=False,
        a=0.1,
        n=1,
        l=None,
        new=False,
        lim=None,
        ):
        assert (up or down or new) == True

        self.add2history()
        tmp = []
        tmp1 = {}

        for gtf in self.gtfs:
            if gtf.l in tmp1:
                tmp1[gtf.l].append(gtf)
            else:
                tmp1[gtf.l] = [gtf]

        if new:
            assert n == 1, 'Currently only n = 1 is supported!'
            assert l not in set([gtf.l for gtf in self.gtfs]), \
                'This l = %d already present, use up or down option!' % l
            if lim:
                tmp1[l] = [GTF(l, [PGTF(a, 1.0, lim=lim)])]
            else:
                tmp1[l] = [GTF(l, [PGTF(a, 1.0)])]

        tmp1k = tmp1.keys()
        tmp1k.sort()

        for key in tmp1k:
            if isinstance(l, int) and l != key:
                tmp.extend(tmp1[key])
                continue

            tmp3 = [i for i in tmp1[key] if len(i.pgtfs) > 1]
            tmp2 = [i for i in tmp1[key] if len(i.pgtfs) == 1]

            t = [i.pgtfs[0].a for i in tmp2]
            tmin = min(t)
            tmax = max(t)

            if up:
                for i in range(n):
                    tmp2.insert(0, GTF(key, [PGTF(tmax * 2 * (i + 1),
                                1.0)]))
            if down:
                for i in range(n):
                    ta = tmin / (2.0 * (i + 1))
                    if ta >= a:
                        tmp2.append(GTF(key, [PGTF(ta, 1.0)]))

            tmp.extend(tmp3 + tmp2)

        self.gtfs = tmp

        return self

    def unmark(self):
        self.add2history()

        for gtf in self.gtfs:
            if len(gtf.pgtfs) == 1:
                gtf.pgtfs[0].opt = False

        return self

    def cutdiff(
        self,
        a=0.1,
        savec=False,
        reverse=False,
        l=None,
        ):
        self.add2history()

        if not savec:
            tmp = []
            for gtf in self.gtfs:
                if len(gtf.pgtfs) > 1:
                    tmp.append(gtf)
                else:

                    if reverse:
                        if not l:
                            if gtf.pgtfs[0].a <= a:
                                tmp.append(gtf)
                        else:
                            if gtf.l != l:
                                tmp.append(gtf)
                            else:
                                if gtf.pgtfs[0].a <= a:
                                    tmp.append(gtf)
                    else:

                        if not l:
                            if gtf.pgtfs[0].a >= a:
                                tmp.append(gtf)
                        else:
                            if gtf.l != l:
                                tmp.append(gtf)
                            else:
                                if gtf.pgtfs[0].a >= a:
                                    tmp.append(gtf)
            self.gtfs = tmp

        else:
            tmp = []

            for gtf in self.gtfs:
                tmp1 = []
                for pgtf in gtf.pgtfs:
                    if reverse:
                        if pgtf.a <= a:
                            tmp1.append(pgtf)
                    else:
                        if pgtf.a >= a:
                            tmp1.append(pgtf)

                if tmp1:
                    gtf.pgtfs = tmp1
                    tmp.append(gtf)

            self.gtfs = tmp

        return self

    def cutcoeff(self, o=2):
        self.add2history()

        tmp = []

        for gtf in self.gtfs:
            if gtf.l != 1 and len(gtf.pgtfs) > 1:

                tmp1 = []
                for pgtf in gtf.pgtfs:
                    if math.fabs(math.log10(math.fabs(pgtf.c[0]))) <= o:
                        tmp1.append(pgtf)
                if tmp1:
                    gtf.pgtfs = tmp1
                    tmp.append(gtf)
            else:

                tmp.append(gtf)

        self.gtfs = tmp

        return self

    def removedups(self):
        self.add2history()

        tmp = []
        tmp1 = {}

        for gtf in self.gtfs:
            assert len(gtf.pgtfs) == 1
            if gtf.l in tmp1:
                tmp1[gtf.l].append(gtf)
            else:
                tmp1[gtf.l] = [gtf]

        tmp1k = tmp1.keys()
        tmp1k.sort()

        for key in tmp1k:
            tmp2 = []
            t = set([i.pgtfs[0].a for i in tmp1[key]])
            for i in t:
                tmp2.append([j for j in tmp1[key] if j.pgtfs[0].a
                            == i][0])

            tmp2.sort(key=lambda x: x.pgtfs[0].a, reverse=True)
            tmp.extend(tmp2)

        self.gtfs = tmp

        return self

    def convert2sp(self):

        self.add2history()

        tmp = []

        for gtf in self.gtfs:
            assert len(gtf.pgtfs) == 1
            if gtf.l == 0:
                gtf.pgtfs[0].c = [1.0, 1.0]
                tmp.append(GTF(1, gtf.pgtfs, gtf.f, gtf.ecp))
            elif gtf.l == 2:
                gtf.pgtfs[0].c = [1.0, 1.0]
                tmp.append(GTF(1, gtf.pgtfs, gtf.f, gtf.ecp))
            else:
                tmp.append(gtf)

        self.gtfs = tmp
        self.removedups()

        return self

    def decontract(
        self,
        l=None,
        n=None,
        diff=None,
        skip=[],
        split=None,
        ):
        """
        l - decontract all functions with given L
        n - number of function with given L to decontract
        diff - decontract only functions with diffuse exponents < diff
        skip - listing with numbers of functions to skip
        split - extract from contraction funcs with exps < split
        """
        self.add2history()
        tmp = []
        t = [i for i in self.gtfs if i.l == l]

        for gtf in self.gtfs:

            if len(gtf.pgtfs) == 1:
                tmp.append(gtf)

            else:

                if (l or l == 0) and n:
                    if gtf.l == l and t:
                        if t.index(gtf) + 1 == n:
                            for pgtf in gtf.pgtfs:
                                tmp.append(GTF(gtf.l, [PGTF(pgtf.a,
                                        [1.0 for c in pgtf.c])], gtf.f,
                                        gtf.ecp))
                    else:
                        tmp.append(gtf)
                elif diff:

                    if [pgtf for pgtf in gtf.pgtfs if pgtf.a < diff]:
                        for pgtf in gtf.pgtfs:
                            tmp.append(GTF(gtf.l, [PGTF(pgtf.a, [1.0
                                    for c in pgtf.c])], gtf.f, gtf.ecp))
                    else:
                        tmp.append(gtf)
                elif split:

                    st = [pgtf for pgtf in gtf.pgtfs if pgtf.a < split]
                    if st:
                        tmp1 = []
                        for pgtf in st:
                            tmp1.append(gtf.pgtfs.pop(gtf.pgtfs.index(pgtf)))
                        tmp.append(gtf)
                        for pgtf in tmp1:
                            tmp.append(GTF(gtf.l, [PGTF(pgtf.a, [1.0
                                    for c in pgtf.c])], gtf.f, gtf.ecp))
                    else:
                        tmp.append(gtf)
                elif skip:

                    if self.gtfs.index(gtf) + 1 in skip:
                        tmp.append(gtf)
                    else:
                        for pgtf in gtf.pgtfs:
                            tmp.append(GTF(gtf.l, [PGTF(pgtf.a, [1.0
                                    for c in pgtf.c])], gtf.f, gtf.ecp))
                else:

                    for pgtf in gtf.pgtfs:
                        tmp.append(GTF(gtf.l, [PGTF(pgtf.a, [1.0
                                   for c in pgtf.c])], gtf.f, gtf.ecp))

        self.gtfs = tmp

        return self

    def crystal_input(self, electrons=[]):

        output = ''
        nelec = 0
        self.electrons = 0

        for i in self.gtfs:
            if not electrons:
                continue

            # NB by EB: warning, below might fail
            if electrons[1] == 0:
                if i.l == 0 or i.l == 1:
                    if electrons[0] >= 2:
                        nelec = 2
                        electrons[0] -= nelec
                    elif electrons[0] > 0:
                        nelec = electrons[0]
                        electrons[0] -= nelec
                    else:
                        nelec = 0
                if i.l == 1 or i.l == 2:
                    if electrons[2] >= 6:
                        nelec = 6
                        electrons[2] -= nelec
                    elif electrons[2] > 0:
                        nelec = electrons[2]
                        electrons[2] -= nelec
                    else:
                        nelec = 0
                if i.l == 3:
                    if electrons[i.l] >= 10:
                        nelec = 10
                        electrons[i.l] -= nelec
                    elif electrons[i.l] > 0:
                        nelec = electrons[i.l]
                        electrons[i.l] -= nelec
                    else:
                        nelec = 0
                if i.l == 4:
                    if electrons[i.l] >= 14:
                        nelec = 14
                        electrons[i.l] -= nelec
                    elif electrons[i.l] > 0:
                        nelec = electrons[i.l]
                        electrons[i.l] -= nelec
                    else:
                        nelec = 0

            elif electrons[i.l] != 0:
                if i.l == 0 and electrons[i.l] >= 2:
                    nelec = 2
                    electrons[i.l] -= 2
                elif i.l == 1 and electrons[i.l] >= 8:
                    nelec = 8
                    electrons[i.l] -= 8
                elif i.l == 2 and electrons[i.l] >= 6:
                    nelec = 6
                    electrons[i.l] -= 6
                elif i.l == 3 and electrons[i.l] >= 10:
                    nelec = 10
                    electrons[i.l] -= 10
                elif i.l == 4 and electrons[i.l] >= 14:
                    nelec = 14
                    electrons[i.l] -= 14
                elif electrons[i.l] > 0:
                    nelec = electrons[i.l]
                    electrons[i.l] -= nelec
                else:
                    nelec = 0

            else:
                nelec = 0

            self.electrons += nelec
            output += str(i.crystal_input(nelec)) + '\n'

        return output.rstrip()

    def gaussian_input(self):
        output = ''

        for i in self.gtfs:
            output += str(i.gaussian_input()) + '\n'

        output += '****\n'

        return output


class BasisSet(object):

    def __init__(
        self,
        no,
        basis=None,
        ecp=None,
        electrons=[],
        nat=None,
        ecp_hack=None,
        ):

        self.descr = chemical_symbols[no]
        self.no = no
        self.basis = basis
        self.ecp = ecp
        self.electrons = electrons

        if not self.electrons:
            z = self.no
            core = 0
            if self.ecp:
                assert self.ecp.core == self.no - self.ecp.z, \
                    'BUG: ECP charge is inconsistent!'
                core = self.ecp.core
            self.electrons = self.__get_electrons(z, core)

        self.nat = nat
        self.ecp_hack = ecp_hack

        if self.nat is None:
            if self.ecp:
                self.nat = 200 + self.no
            elif self.ecp_hack:
                self.nat = 200 + self.no
            else:
                self.nat = self.no

    def __get_electrons(self, z, core=0):
        """
        returns distribution of electrons per s,p,d,f shells for given Z
        Order of shell filling from Wikipedia
        """
        electrons = [0, 0, 0, 0, 0] # SP shell included
        electrons_core = [0, 0, 0, 0, 0]

        max_el = { # '1' stands for SP shell
            0: 2,
            1: 0,
            2: 6,
            3: 10,
            4: 14,
        }

        fill_z = [
            (0, '1s'),
            (0, '2s'),
            (1, '2p'),
            (0, '3s'),
            (1, '3p'),
            (0, '4s'),
            (2, '3d'),
            (1, '4p'),
            (0, '5s'),
            (2, '4d'),
            (1, '5p'),
            (0, '6s'),
            (3, '4f'),
            (2, '5d'),
            (1, '6p'),
            (0, '7s'),
            (3, '5f'),
            (2, '6d'),
            (1, '7p'),
            (0, '8s'),
        ]

        fill_core = [
            (0, '1s'),
            (0, '2s'),
            (1, '2p'),
            (0, '3s'),
            (1, '3p'),
            (2, '3d'),
            (0, '4s'),
            (1, '4p'),
            (2, '4d'),
            (3, '4f'),
            (0, '5s'),
            (1, '5p'),
            (2, '5d'),
            (3, '5f'),
            (0, '6s'),
            (1, '6p'),
            (2, '6d'),
            (0, '7s'),
            (1, '7p'),
            (0, '8s'),
        ]

        def set_l(zz, l):
            if l > 0:
                l += 1
            nelec = max_el[l]
            if zz >= nelec:
                electrons[l] += nelec
            elif zz > 0:
                electrons[l] += zz
                nelec = zz
            else:
                nelec = 0
            zz -= nelec
            return zz

        def remove_l(zcore, l):
            if l > 0:
                l += 1
            nelec = max_el[l]
            if zcore > 0:
                if zcore >= nelec:
                    electrons_core[l] += nelec
                    electrons[l] -= nelec
                else:
                    nelec = zcore
                    electrons_core[l] += nelec
                    electrons[l] -= nelec
                zcore -= nelec
            return zcore

        zz = z # zz will be decreased during electron distribution
        zcore = core # core electrons
        for item in fill_z:
            zz = set_l(zz, item[0])
            logging.debug('(all electron) %s: %s' % (item[1], electrons))

        # fix some cases for full d- and f-shells

        if electrons[3] % 10 == 4 or electrons[3] % 10 == 9:
            electrons[0] -= 1
            electrons[3] += 1
        if electrons[4] % 14 == 6 or electrons[4] % 14 == 13:
            electrons[0] -= 1
            electrons[4] += 1

        # remove core electrons

        for item in fill_core:
            zcore = remove_l(zcore, item[0])
            logging.debug('(core) %s: %s %s' % (item[1], electrons, electrons_core))
        logging.debug('electrons_core: %s' % electrons_core)
        logging.debug('electrons: %s' % electrons)
        return electrons

    def __repr__(self):
        output = [self.descr.lower().capitalize(), self.no]

        if self.basis:
            output.append(self.basis)
        if self.ecp:
            output.append(self.ecp)

        return pformat(output)

    def crystal_input(self, electrons=[]):
        if electrons:
            self.electrons = electrons

        output = ''

        if self.ecp:
            output += ' '.join([str(200 + self.no),
                               str(len(self.basis.gtfs))]) + '\n'
        elif self.nat:
            output += ' '.join([str(self.nat),
                               str(len(self.basis.gtfs))]) + '\n'
        elif self.ecp_hack:
            output += ' '.join([str(200 + self.no),
                               str(len(self.basis.gtfs))]) + '\n'
        else:
            output += ' '.join([str(self.no),
                               str(len(self.basis.gtfs))]) + '\n'
        if self.ecp:
            output += self.ecp.crystal_input() + '\n'
        if self.ecp_hack:
            output += self.ecp_hack + '\n'

        output += \
            self.basis.crystal_input(electrons=copy.copy(self.electrons)) + '\n'

#        if self.ecp:
#            if self.basis.electrons == int(self.basis.electrons):
#                assert self.ecp.z == self.basis.electrons
#        elif self.ecp_hack:
#            pass
#        else:
#            if self.basis.electrons == int(self.basis.electrons):
#                assert self.no == self.basis.electrons

        return output

    def gaussian_input(self):
        output = '%s%5d\n' % (self.descr, 0)
        output += str(self.basis.gaussian_input())
        output += '\n'

        if self.ecp:
            output += '%s%5d\n' % (self.descr, 0)
            output += str(self.ecp.gaussian_input())
        output += '\n'

        return output


def parse_bs(text):
    """
    Parsing basis + ECP files in Gaussian'94 format
    """
    atomstr = {}
    for k in chemical_symbols:
        atomstr[k.upper()] = atomic_numbers[k]

    atoms = [] # storing results here

    # Parsing File
    results_raw = grammar.parseString(text)

    if len(results_raw) == 2:
        results_list = zip(results_raw[0], results_raw[1])
    else:
        results_list = results_raw[0]

    # Results, if any
    for results_item in results_list:
        results = results_item[0]

        # FIXME fails in Py3 on some basis sets
        logging.debug('bdescr: %s' % results.bdescr)
        atom = results.bdescr
        gtfs = []

        for shell in results.basis:
            linfo = shell[0]
            exps = shell[1:]

            assert linfo[1] == len(exps)

            l = lstr[linfo[0]]
            pgtfs = []

            for exp in exps:
                pgtfs.append(PGTF(exp[0], exp[1:]))

            gtfs.append(GTF(l, pgtfs, f=linfo[-1]))

        basis = Basis(gtfs)
        ecp = None

        if len(results_item) > 1:
            results = results_item[1]

            logging.debug('edescr: %s' % results.edescr)
            logging.debug('lmax: %s' % results.lmax)
            logging.debug('core: %s' % results.core)

            assert atom == results.edescr, \
                'Oops! ECP and Basis for different elements!'
            assert results.lmax + 1 == len(results.ecp), \
                'Oops! Wrong number of components in ECP!'

            gtfs = []
            llist = [results.lmax] + [i for i in range(results.lmax)]

            assert len(llist) == len(results.ecp)

            for i in range(len(llist)):
                l = llist[i]
                n = results.ecp[i][0]
                exps = (results.ecp[i])[1:]
                assert n == len(exps)
                pgtfs = []
                for exp in exps:
                    pgtfs.append(PGTF(exp[1], exp[2], exp[0]))
                gtfs.append(GTF(l, pgtfs, ecp=True))

            ecp = ECP(results.lmax, results.core,
                      atomstr[results.edescr] - results.core, gtfs)

        atoms.append(BasisSet(atomstr[atom], basis, ecp))

    return tuple(atoms)
