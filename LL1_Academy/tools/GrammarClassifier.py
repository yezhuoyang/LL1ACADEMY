# -*- coding: utf-8 -*-
# Matt's attempt at a SVM classifier for identifying interesting
# grammars.

from enum import Enum
import string
import logging
import random
import numpy as np
from functools import reduce

epsilon = u'ε'

# Production types:
# 1. Epsilon
# 2. All terminals
# 3. Nonterminal.
# 4. Terminals and 1 nonterminal
# 5. Terminals and 2 nonterminals
# 6. Terminals and 3 nonterminals

# It's probably a good idea to test for edge cases, and make those a
# feature.
#
# For follow sets:
# 1) if a nonterminal A occurs as the last
#    character a production for B, then follow(A) should include
#    follow(B).
# 2) if a production includes adjacent nonterminals AB, then
#    follow(A) should include first(B).
#
# For first sets:
# 1) Left-recursion
# 2) Nonterminal in leftmost position of a production
#
# For nullable:
# 1) Epsilon production
# 2) Production of all nullable nonterminals
# 3) Left-recursion
#
# For exam generation, we want to make sure the questions we generate
# cover all the edge cases. How can we achieve that?
#
# Rather than randomly generating a grammar and then testing, we may
# simply generate variants of a set of prototypical grammars. This
# ensures we have one similar to each prototype.
# Each variant changes inessential properties:
# - Change the terminal symbols (and number)
# - Rename nonterminals
# - Change the order of productions
# - Add noise: benign productions and nonterminals (that don't affect nullability, LL1-ness, etc)
#   - E.g. a production of all terminals.
#
# We can specify a prototypical grammar using wildcards as follows:
# - * denotes zero or more terminal symbols
# - + denotes one or more terminal symbols
#
# A -> *B+C* | BC* | +C
# 

ProductionType = Enum(
  'ProductionType',
  'EPSILON ALL_TERMS NONTERM TERMS_1_NONTERM TWO_NONTERMS THREE_NONTERMS'
  )

# concatenate a list of lists
def concatList(ls):
  return reduce(lambda x, y: x + y, ls, [])

# return a list of the upper-case letters in a string
def uppers(s):
  return [c for c in s if c in string.ascii_uppercase]

def lowers(s):
  return [c for c in s if c in string.ascii_lowercase]

logger = logging.getLogger('GrammarAnalyzer')

class GrammarAnalyzer:
  """
  Analyzes a grammar and returns a feature set.

  Usage: new GrammarAnalyzer(g).run()
  """
  
  def __init__(self,g):
    self.g = g
    
    # all the terminals that appear in the grammar
    self.terminals = list(set(lowers(''.join([p for nt in g for p in g[nt]]))))

    self.firstSets = {}
    self.followSets = {}
    self.parseTable = {}

    # initialize first and follow sets
    for nt in g:
      self.firstSets[nt] = set()
      self.followSets[nt] = set()
      self.parseTable[nt] = {}
    self.nullable = set()
    self.isLL1 = True
    
    self.compute_nullable()
    self.compute_firstSets()
    self.compute_followSets()
    self.compute_parseTable()

    # compute first and follow constraints
    self.firstSet_constraints()
    self.followSet_constraints()
    

  def compute_nullable(self):
    # mark nt nullabe, return True if nullable table is changed
    def mark(nt):
      if nt in self.nullable:
        return False
      else:
        self.nullable.add(nt)
        return True

    def symbol_nullable(s):
      if s == epsilon:
        return True
      if s in string.ascii_uppercase:
        return s in self.nullable
      return False
      
    change = True
    while(change):
      change = False
      for nt in self.g:
        for p in self.g[nt]:
          if all([symbol_nullable(s) for s in p]):
            change = mark(nt) or change

  def first_of_production(self,p):
    if p == epsilon or len(p) == 0:
      return set()
    if p[0] in string.ascii_lowercase:
      return set(p[0])
      
    s = set()
    for x in p:
      if x in string.ascii_lowercase:
        # terminal
        s.add(x)
        return s

      # nonterminal
      s.update(self.firstSets[x])
          
      if x not in self.nullable:
        return s
    return s
            
  def compute_firstSets(self):
    change = True
    while(change):
      change = False
      for nt in self.g:
        oldLen = len(self.firstSets[nt])
        for p in self.g[nt]:
          self.firstSets[nt].update(self.first_of_production(p))
          
        if oldLen < len(self.firstSets[nt]):
          change = True

  def firstSet_constraints(self):
    # collect constraints
    constraints = set()
    for nt in self.g:
      for p in self.g[nt]:
        if p == epsilon or len(p) == 0:
          continue
        for c in p:
          if c in string.ascii_lowercase:
            constraints.add((c, 'in', nt))
            break
          elif c in string.ascii_uppercase:
            constraints.add((c, 'subset', nt))
            if c not in self.nullable:
              break

    # solve constraints
    firstSets = {}
    for nt in self.g:
      firstSets[nt] = set()

    # returns True if added new elements
    def update(nt,s):
      l1 = len(firstSets[nt])
      firstSets[nt].update(s)
      if l1 < len(firstSets[nt]):
        return True
      else:
        return False

    changed = True
    while changed:
      changed = False
      for c in constraints:
        if c[1] == 'in':
          changed = update(c[2],set(c[0])) or changed
        elif c[1] == 'subset':
          changed = update(c[2],firstSets[c[0]])

    self.firstConstraints = constraints
    return firstSets

  def firstConstraintCycles(self):
    def subsetsOf(x):
      todo = [x]
      found = []
      while len(todo) > 0:
        nt = todo.pop()
        for c in self.firstConstraints:
          if c[1] == 'subset':
            if c[2] == nt:
              # c[0] is a subset of nt
              if c[0] not in found:
                #print '%s is subset of %s' % (c[0], nt)
                found.append(c[0])
                todo.append(c[0])
      return found

    answer = False
    for nt in self.g:
      if nt in subsetsOf(nt):
        answer = True

    #if answer:
    #  print "Grammar with firstConstraintCycles: %s" % self.g
      
    return answer

  
  def followSet_constraints(self):
    constraints = set()
    def splitProd(p):
      for i in range(len(p)):
        if p[i] in string.ascii_uppercase:
          yield (p[:i],p[i],p[i+1:])

    for nt in self.g:
      for p in self.g[nt]:
        for (s1,x,s2) in splitProd(p):
          for c in self.first_of_production(s2):
            constraints.add((c, 'in', x))
          if self.production_nullable(s2):
            # whatever can follow nt can also follow x (when s2 is null)
            constraints.add((nt, 'subset', x))

    followSets = {}
    for nt in self.g:
      followSets[nt] = set()
    followSets['A'].add('$')

    def update(nt,s):
      l1 = len(followSets[nt])
      followSets[nt].update(s)
      if l1 < len(followSets[nt]):
        return True
      else:
        return False

    changed = True
    while changed:
      changed = False
      for c in constraints:
        if c[1] == 'in':
          changed = update(c[2],c[0]) or changed
        elif c[1] == 'subset':
          changed = update(c[2],followSets[c[0]]) or changed

    self.followConstraints = constraints
    return followSets

  def followConstraintCycles(self):
    def subsetsOf(x):
      todo = [x]
      found = []
      while len(todo) > 0:
        nt = todo.pop()
        for c in self.followConstraints:
          if c[1] == 'subset':
            if c[2] == nt:
              # c[0] is a subset of nt
              if c[0] not in found:
                # print '%s is subset of %s' % (c[0], nt)
                found.append(c[0])
                todo.append(c[0])
      return found

    answer = False
    for nt in self.g:
      if nt in subsetsOf(nt):
        answer = True
    return answer
        
  def compute_followSets(self):
    change = True
    self.followSets['A'].add('$')
    logger.debug('Computing followSets for %s' % self.g)
    
    def updateFollow(nt, c):
      s = c
      if not s.issubset(self.followSets[nt]):
        logger.debug("Adding %s to follow(%s) = %s" % (s,nt,self.followSets[nt]))
        self.followSets[nt].update(s)
        return True
      return False

    def splitProd(p):
      for i in range(len(p)):
        if p[i] in string.ascii_uppercase:
          yield (p[:i],p[i],p[i+1:])

    while(change):
      change = False
      for nt in self.g:
        for p in self.g[nt]:
          for (p1,x,p2) in splitProd(p):
            change = updateFollow(x, self.first_of_production(p2)) or change
            if self.production_nullable(p2):
              change = updateFollow(x, self.followSets[nt]) or change

  def production_nullable(self,p):
    if p == epsilon:
      return True
    for s in p:
      if s in string.ascii_lowercase:
        return False
      if s not in self.nullable:
        return False
    return True
            
  def compute_parseTable(self):
    def addTransition(nt,t,prod):
      if t not in self.parseTable[nt]:
        self.parseTable[nt][t] = set([prod])
      else:
        self.isLL1 = False
        self.parseTable[nt][t].add(prod)
        
    for nt in self.g:
      for p in self.g[nt]:
        for t in self.first_of_production(p):
          addTransition(nt,t,p)
        if self.production_nullable(p):
          for t in self.followSets[nt]:
            addTransition(nt,t,p)

  def run(self):
    nt_features = dict([(nt, self.nonterminal_features(g,nt)) for nt in g])

    # sort nonterminals by their features
    nts = list(g.keys())
    nts.sort(lambda k1, k2: cmp(nt_features[k1], nt_features[k2]))

    # concatentate all features
    return concat([nt_features[nt] for nt in nts])
    
  def nonterminal_features(self, nt):
    """Format:

    We include a number of features per nonterminal:
    
    - Number of productions
    - Production variety: how many different types of productions are there?
    - Recursive (directly or indirectly)
    - Number of epsilon productions
    - Number of all terminal productions
    - Number of nonterminal productions
    - Number of terminals and 1 nonterminal productions
    - Number of terminals and 2 nonterminal productions
    - Number of terminals and 3 nonterminal productions
    - Fraction of terminals that appear in the first set
    - Fraction of terminals that appear in the follow set

    Each nonterminal has 11 features. We normalize by sorting the
    nonterminals by the feature sets (lexicographically, so the number
    of productions is most important, etc). Then concatenate the
    features of each nonterminal to obtain the features of the entire
    grammar.

    """

    productions = self.g[nt]
    production_types = [self.production_type(p) for p in productions]
    recursive = self.reachable(nt,nt)
    prod_type_counts = Counter(production_types)
    
    return [
      len(productions),               # number of productions
      len(set(production_types)),     # number of different production types
      recursive,
      prod_type_counts[ProductionType.EPSILON],
      prod_type_counts[ProductionType.ALL_TERMS],
      prod_type_counts[ProductionType.NONTERM],
      prod_type_counts[ProductionType.TERMS_1_NONTERM],
      prod_type_counts[ProductionType.TWO_NONTERMS],
      prod_type_counts[ProductionType.THREE_NONTERMS],
      len(self.firstSets[nt])/len(self.terminals),
      len(lowers(self.followSets[nt]))/len(self.terminals) # lowers: remove $ from follow set
      ]

  # no nt is nullable
  def countNullable(self):
    return len(self.nullable)
  
  # test if all first sets are empty
  def allFirstsEmpty(self):
    for nt in self.g:
      if len(self.firstSets[nt]) > 0:
        return False
    return True

  # test if any first sets are empty
  def anyFirstsEmpty(self):
    for nt in self.g:
      if len(self.firstSets[nt]) == 0:
        return True
    return False
  
  def sameFirstSets(self):
    s = None
    for nt in self.g:
      if s == None:
        s = self.firstSets[nt]
      if self.firstSets[nt] != s:
        return False
    return True

  def sameFollowSets(self):
    s = None
    for nt in self.g:
      if s == None:
        s = self.firstSets[nt]
      if self.firstSets[nt] != s:
        return False
    return True
  
  # all rules have at most one production
  def noChoice(self):
    for nt in self.g:
      if len(self.g[nt]) > 1:
        return False
    return True

  def countOneProductionRules(self):
    c = 0
    for nt in self.g:
      if len(self.g[nt]) == 1:
        c = c + 1
    return c

    
  def all_reachable(self):
    # assume 'A' is the start symbol
    for nt in self.g:
      if nt == 'A':
        continue
      if not self.reachable(nt, 'A'):
        return False
    
    return True
    
  def reachable(self, ntDest, ntSrc):
    return self.reachable_path(ntDest, [ntSrc])

  def reachable_path(self, ntDest, ntPath):
    return any([self.reachable_prod(ntDest, ntPath, p) for p in self.g[ntPath[-1]]])

  def reachable_prod(self, ntDest, ntPath, prod):
    if ntDest in uppers(prod):
      return True
    else:
      return any([self.reachable_path(ntDest, ntPath + [nt])
                  for nt in uppers(prod)
                  if nt not in ntPath])

  def production_features(self, nt, p):
      pass

  def production_type(p):
    if(p == epsilon):
      return ProductionType.EPSILON

    nts = length(uppers(p))

    if(nts == 0):
      return ProductionType.ALL_TERMS

    if(length(p) == 1):
      return ProductionType.NONTERM

    if(nts == 1):
      return ProductionType.TERMS_1_NONTERM

    if(nts == 2):
      return ProductionType.TWO_NONTERM

    if(nts == 3):
      return ProductionType.THREE_NONTERM

    raise Exception('cannot determine production type for ' + p)


def generateProduction(nts, ts, maxProdLen):
  # print "generateProduction"
  syms = nts + ts # make terminals twice as frequent as nonterminals
  len = random.randint(0,maxProdLen)
  r = []
  for i in range(len):
    r.append(random.choice(syms))
  return ''.join(r)

def generateRule(nt, nts, ts, maxProds, maxProdLen):
  # print "generateRule(%s)" % nt
  r = []
  prods = random.randint(1,maxProds)
  while len(r) < prods:
    p = generateProduction(nts, ts, maxProdLen)

    # don't allow A -> A
    if p == nt:
      continue

    # don't allow duplicates
    if p in r:
      continue
      
    r.append(p)
  return r


def generateGrammar(nts, ts, maxProds, maxProdLen):
  g = {}
  for nt in nts:
    g[nt] = None
    while g[nt] == None:
      r = generateRule(nt, nts, ts, maxProds, maxProdLen)

      # make sure it can generate some string
      if r == [] or r == ['']:
        continue
      if not any([c in string.ascii_lowercase for p in r for c in p]):
        continue
      if not any([len(p) > 0 and p[0] in string.ascii_lowercase for p in r]):
        continue
      
      # not all productions can have self-reference
      if all([nt in p for p in r]):
        continue

      # looks good; let's keep it
      g[nt] = r
      
        
  return g

# Predicate DSL
class AND:
  def __init__(self,*ps):
    self.ps = ps
  def __call__(self,a):
    return all([p(a) for p in self.ps]) #self.p1(a) and self.p2(a)
  def __str__(self):
    # return "%s and %s" % (self.p1, self.p2)
    return ' and '.join([str(p) for p in self.ps])

class ExistsNTWithOneProduction:
  def __call__(self,a):
    return any([len(a.g[nt]) == 1 for nt in a.g])
  def __str__(self):
    return 'ExistsNTWithOneProduction'

class ExistsNullableNT:
  def __call__(self,a):
    return len(a.nullable) > 0
  def __str__(self):
    return 'ExistsNullableNT'
  
class NOT:
  def __init__(self, p):
    self.p = p
  def __call__(self, a):
    return not self.p(a)
  def __str__(self):
    return "NOT(%s)" % self.p
  
class isLL1:
  def __call__(self,a):
    return a.isLL1
  def __str__(self):
    return "isLL1"

class FollowConstraintCycles:
  def __call__(self,a):
    return a.followConstraintCycles()
  def __str__(self):
    return "followConstraintCycles"

class FirstConstraintCycles:
  def __call__(self,a):
    return a.firstConstraintCycles()
  def __str__(self):
    return "firstConstraintCycles"

class NoNullableNTs:
  def __call__(self,a):
    return a.countNullable() == 0
  def __str__(self):
    return "NoNullableNTs"

grammarSpecs = {
    'EASY_LL1' :
    AND(isLL1(), NOT(FollowConstraintCycles())),
    
    'EASY_NON_LL1' :
    AND(NOT(isLL1()),
        NOT(FollowConstraintCycles()),
        NOT(FirstConstraintCycles())),
    
    'MED_LL1' :
    AND(isLL1(), FollowConstraintCycles()),

    # Left-recursion
    'MED1_NON_LL1' :
    AND(NOT(isLL1()),
        NOT(FollowConstraintCycles()),
        FirstConstraintCycles()),

    # Left-recursion
    'MED2_NON_LL1' :
    AND(NOT(isLL1()),
        FollowConstraintCycles(),
        NOT(FirstConstraintCycles())),
    
    'HARD_LL1' :
    AND(isLL1(),
        ExistsNullableNT(),
        NOT(ExistsNTWithOneProduction()),
        FollowConstraintCycles()),
    
    'HARD_NON_LL1' :
    AND(NOT(isLL1()),
        ExistsNullableNT(),
        NOT(ExistsNTWithOneProduction()),
        FollowConstraintCycles(),
        FirstConstraintCycles())
  }  
def findGrammar(pred, **kwargs):
  pred = grammarSpecs.get(pred, pred)
  
  gen = kwargs.get('gen',generateGrammar)
  nts = kwargs.get('nonterminals', 'ABC')
  ts = kwargs.get('terminals', 'wxyz')
  maxProds = kwargs.get('maxProds', 3)
  maxProdLen = kwargs.get('maxProdLen', 3)
  
  n = 0
  while n < 10000:
    n = n + 1
    g = gen(nts, ts, maxProds, maxProdLen)
    a = GrammarAnalyzer(g)
    
    # make sure it's not trivial
    if not a.all_reachable():
      continue
    if a.anyFirstsEmpty():
      continue
    if a.countOneProductionRules() > 1:
      continue
    if a.sameFirstSets() or a.sameFollowSets():
      continue
    
    if pred(a):
      #print "Found grammar in %d steps" % n
      return g


# find 2 grammars that match pred: one LL1 and one not    
def findGrammars(pred):
  return (findGrammar(lambda a: a.isLL1 and pred(a)),
          findGrammar(lambda a: not a.isLL1 and pred(a)))
    
# TODO: it seems that randomly generating LL1 grammars can be hard.
# Can we reverse engineer one? I.e. generate a random parse table.
# Then find a grammar the implements it.
#
# This can be generalized to work for both LL1 and non-LL1 grammars
# To achieve non-LL1, first generate an LL1 table and then insert an extra
# production

class ParseTableGenerator:
  def __init__(self, nts, ts, maxProds, maxProdLen):
    self.nts = [nt for nt in nts]
    self.ts = [t for t in ts]
    self.maxProds = maxProds
    self.maxProdLen = maxProdLen
    
    self.g = {}
    self.tbl = {}
    for nt in nts:
      self.g[nt] = []
      self.tbl[nt] = {}

    self.initNullable()
    #self.initFirstCycle()
    self.randomParseTable()

  def initFirstCycle(self):
    [nt1,nt2] = list(np.random.choice(self.nts, 2, replace=False))
    self.g[nt1].append(nt2)
    self.g[nt2].append(nt1)
    
  def newProduction(self, nt, firstTerm):
    # reanalyze each time, as we are adding production
    a = GrammarAnalyzer(self.g)
    nts = filter(lambda nt: firstTerm in a.firstSets[nt], self.nts)

    fails = 0
    while fails < 3:
      first = random.choice([firstTerm] + nts)

      # sometimes add a nullable nonterminal
      if random.choice([True,False]):
        nullables = [null for null in self.nts
                     if (null is not nt and
                         null in a.nullable)]
        if len(nullables) > 0: 
          first = random.choice(nullables) + first
      
      p = first + generateProduction(self.nts, self.ts, self.maxProdLen-len(first))
      if p not in self.g[nt]:
        #self.tbl[nt][firstTerm] = [p]
        self.g[nt].append(p)

        # adding this new rule could impact the parse table in other
        # ways e.g. if p starts with a non-terminal, could change parseTable
        # for nt on other terminals too. Check for this
        a1 = GrammarAnalyzer(self.g)
        if not a1.isLL1:
          #print "removing production: %s" % p
          self.g[nt].remove(p)
          fails = fails + 1
        else:
          self.tbl = a1.parseTable
          return True

    # print "failed to add production"
    # failed
    return False

  def initNullable(self):
    epsilons = random.choice([1,1,1,2,2,3])
    nts = np.random.choice(self.nts, epsilons, replace=False)
    for nt in nts:
      self.g[nt].append('')
    
  def randomParseTableOld(self):
    # For each entry, choose between Error and something.  If something,
    # generate a production. Helper: generate a production that
    # generates strings starting with given terminal. Input: the grammar
    # so far, and the terminal symbol. Will consider the first sets so
    # far when

    for nt in self.nts:
      for t in self.ts:
        if t not in self.tbl[nt]:
          add = random.choice([True,True,False])
          if add:
            self.newProduction(nt,t)

  def randomParseTable(self):
    # choose a random NT/term that is currently empty

    fails = 0
    while fails < 3:
      choices = [(nt,t)
                 for nt in self.nts
                 for t in self.ts
                 if (t not in self.tbl[nt]
                     and
                     len(self.g[nt]) < self.maxProds)]
      if len(choices) == 0:
        break
      (nt,t) = random.choice(choices)

      if self.newProduction(nt,t):
        fails = 0
      else:
        fails = fails+1

def generateLL1Grammar(nts,ts,maxProds,maxProdLen):
  gen = ParseTableGenerator(nts,ts,maxProds,maxProdLen)
  return gen.g

