#!/usr/bin/env python
"""
.. module:: hindley_milner
   :synopsis: An implementation of the Hindley Milner type checking algorithm
              based on the Scala code by Andrew Forrest, the Perl code by
              Nikita Borisov and the paper "Basic Polymorphic Typechecking"
              by Cardelli.
.. moduleauthor:: Robert Smallshire
"""

from __future__ import print_function


# =======================================================#
# Class definitions for the abstract syntax tree nodes
# which comprise the little language for which types
# will be inferred

class Lambda(object):
    """Lambda abstraction"""

    def __init__(self, v, body):
        self.v = v
        self.body = body

    def __str__(self):
        return "(fn {v} => {body})".format(v=self.v, body=self.body)


class Ident(object):
    """Identfier"""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Apply(object):
    """Function application"""

    def __init__(self, fn, arg):
        self.fn = fn
        self.arg = arg

    def __str__(self):
        return "({fn} {arg})".format(fn=self.fn, arg=self.arg)


class Let(object):
    """Let binding"""

    def __init__(self, v, defn, body):
        self.v = v
        self.defn = defn
        self.body = body

    def __str__(self):
        return "(let {v} = {defn} in {body})".format(v=self.v, defn=self.defn, body=self.body)


class Letrec(object):
    """Letrec binding"""

    def __init__(self, v, defn, body):
        self.v = v
        self.defn = defn
        self.body = body

    def __str__(self):
        return "(letrec {v} = {defn} in {body})".format(v=self.v, defn=self.defn, body=self.body)


# =======================================================#
# Exception types

class InferenceError(Exception):
    """Raised if the type inference algorithm cannot infer types successfully"""

    def __init__(self, message):
        self.__message = message

    message = property(lambda self: self.__message)

    def __str__(self):
        return str(self.message)


class ParseError(Exception):
    """Raised if the type environment supplied for is incomplete"""

    def __init__(self, message):
        self.__message = message

    message = property(lambda self: self.__message)

    def __str__(self):
        return str(self.message)


# =======================================================#
# Types and type constructors

class TypeVariable(object):
    """A type variable standing for an arbitrary type.

    All type variables have a unique id, but names are only assigned lazily,
    when required.
    """

    next_variable_id = 0

    def __init__(self):
        self.id = TypeVariable.next_variable_id
        TypeVariable.next_variable_id += 1
        self.instance = None
        self.__name = None

    next_variable_name = 'a'

    def _getName(self):
        """Names are allocated to TypeVariables lazily, so that only TypeVariables
        present
        """
        if self.__name is None:
            self.__name = TypeVariable.next_variable_name
            TypeVariable.next_variable_name = chr(ord(TypeVariable.next_variable_name) + 1)
        return self.__name

    name = property(_getName)

    def __str__(self):
        if self.instance is not None:
            return str(self.instance)
        else:
            return self.name

    def __repr__(self):
        return "TypeVariable(id = {0})".format(self.id)


class TypeOperator(object):
    """An n-ary type constructor which builds a new type from old"""

    def __init__(self, name, types):
        self.name = name
        self.types = types

    def __str__(self):
        num_types = len(self.types)
        if num_types == 0:
            return self.name
        elif num_types == 2:
            return "({0} {1} {2})".format(str(self.types[0]), self.name, str(self.types[1]))
        else:
            return "{0} {1}" % (self.name, ' '.join(self.types))


class Function(TypeOperator):
    """A binary type constructor which builds function types"""

    def __init__(self, from_type, to_type):
        super(Function, self).__init__("->", [from_type, to_type])


# Basic types are constructed with a nullary type constructor
Integer = TypeOperator("int", [])  # Basic integer
Bool = TypeOperator("bool", [])  # Basic bool


# =======================================================#
# Type inference machinery

def analyse(node, env, non_generic=None):
    """Computes the type of the expression given by node.

    The type of the node is computed in the context of the context of the
    supplied type environment env. Data types can be introduced into the
    language simply by having a predefined set of identifiers in the initial
    environment. environment; this way there is no need to change the syntax or, more
    importantly, the type-checking program when extending the language.

    Args:
        node: The root of the abstract syntax tree.
        env: The type environment is a mapping of expression identifier names
            to type assignments.
            to type assignments.
        non_generic: A set of non-generic variables, or None

    Returns:
        The computed type of the expression.

    Raises:
        InferenceError: The type of the expression could not be inferred, for example
            if it is not possible to unify two types such as Integer and Bool
        ParseError: The abstract syntax tree rooted at node could not be parsed
    """

    if non_generic is None:
        non_generic = set()

    if isinstance(node, Ident):
        return getType(node.name, env, non_generic)
    elif isinstance(node, Apply):
        fun_type = analyse(node.fn, env, non_generic)
        arg_type = analyse(node.arg, env, non_generic)
        result_type = TypeVariable()
        unify(Function(arg_type, result_type), fun_type)
        return result_type
    elif isinstance(node, Lambda):
        arg_type = TypeVariable()
        new_env = env.copy()
        new_env[node.v] = arg_type
        new_non_generic = non_generic.copy()
        new_non_generic.add(arg_type)
        result_type = analyse(node.body, new_env, new_non_generic)
        return Function(arg_type, result_type)
    elif isinstance(node, Let):
        defn_type = analyse(node.defn, env, non_generic)
        new_env = env.copy()
        new_env[node.v] = defn_type
        return analyse(node.body, new_env, non_generic)
    elif isinstance(node, Letrec):
        new_type = TypeVariable()
        new_env = env.copy()
        new_env[node.v] = new_type
        new_non_generic = non_generic.copy()
        new_non_generic.add(new_type)
        defn_type = analyse(node.defn, new_env, new_non_generic)
        unify(new_type, defn_type)
        return analyse(node.body, new_env, non_generic)
    assert 0, "Unhandled syntax node {0}".format(type(t))


def getType(name, env, non_generic):
    """Get the type of identifier name from the type environment env.

    Args:
        name: The identifier name
        env: The type environment mapping from identifier names to types
        non_generic: A set of non-generic TypeVariables

    Raises:
        ParseError: Raised if name is an undefined symbol in the type
            environment.
    """
    if name in env:
        return fresh(env[name], non_generic)
    elif isIntegerLiteral(name):
        return Integer
    else:
        raise ParseError("Undefined symbol {0}".format(name))


def fresh(t, non_generic):
    """Makes a copy of a type expression.

    The type t is copied. The the generic variables are duplicated and the
    non_generic variables are shared.

    Args:
        t: A type to be copied.
        non_generic: A set of non-generic TypeVariables
    """
    mappings = {}  # A mapping of TypeVariables to TypeVariables

    def freshrec(tp):
        p = prune(tp)
        if isinstance(p, TypeVariable):
            if isGeneric(p, non_generic):
                if p not in mappings:
                    mappings[p] = TypeVariable()
                return mappings[p]
            else:
                return p
        elif isinstance(p, TypeOperator):
            return TypeOperator(p.name, [freshrec(x) for x in p.types])

    return freshrec(t)


def unify(t1, t2):
    """Unify the two types t1 and t2.

    Makes the types t1 and t2 the same.

    Args:
        t1: The first type to be made equivalent
        t2: The second type to be be equivalent

    Returns:
        None

    Raises:
        InferenceError: Raised if the types cannot be unified.
    """

    a = prune(t1)
    b = prune(t2)
    if isinstance(a, TypeVariable):
        if a != b:
            if occursInType(a, b):
                raise InferenceError("recursive unification")
            a.instance = b
    elif isinstance(a, TypeOperator) and isinstance(b, TypeVariable):
        unify(b, a)
    elif isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
        if (a.name != b.name or len(a.types) != len(b.types)):
            raise InferenceError("Type mismatch: {0} != {1}".format(str(a), str(b)))
        for p, q in zip(a.types, b.types):
            unify(p, q)
    else:
        assert 0, "Not unified"


def prune(t):
    """Returns the currently defining instance of t.

    As a side effect, collapses the list of type instances. The function Prune
    is used whenever a type expression has to be inspected: it will always
    return a type expression which is either an uninstantiated type variable or
    a type operator; i.e. it will skip instantiated variables, and will
    actually prune them from expressions to remove long chains of instantiated
    variables.

    Args:
        t: The type to be pruned

    Returns:
        An uninstantiated TypeVariable or a TypeOperator
    """
    if isinstance(t, TypeVariable):
        if t.instance is not None:
            t.instance = prune(t.instance)
            return t.instance
    return t


def isGeneric(v, non_generic):
    """Checks whether a given variable occurs in a list of non-generic variables

    Note that a variables in such a list may be instantiated to a type term,
    in which case the variables contained in the type term are considered
    non-generic.

    Note: Must be called with v pre-pruned

    Args:
        v: The TypeVariable to be tested for genericity
        non_generic: A set of non-generic TypeVariables

    Returns:
        True if v is a generic variable, otherwise False
    """
    return not occursIn(v, non_generic)


def occursInType(v, type2):
    """Checks whether a type variable occurs in a type expression.

    Note: Must be called with v pre-pruned

    Args:
        v:  The TypeVariable to be tested for
        type2: The type in which to search

    Returns:
        True if v occurs in type2, otherwise False
    """
    pruned_type2 = prune(type2)
    if pruned_type2 == v:
        return True
    elif isinstance(pruned_type2, TypeOperator):
        return occursIn(v, pruned_type2.types)
    return False


def occursIn(t, types):
    """Checks whether a types variable occurs in any other types.

    Args:
        v:  The TypeVariable to be tested for
        types: The sequence of types in which to search

    Returns:
        True if t occurs in any of types, otherwise False
    """
    return any(occursInType(t, t2) for t2 in types)


def isIntegerLiteral(name):
    """Checks whether name is an integer literal string.

    Args:
        name: The identifier to check

    Returns:
        True if name is an integer literal, otherwise False
    """
    result = True
    try:
        int(name)
    except ValueError:
        result = False
    return result


# ==================================================================#
# Example code to exercise the above


def tryExp(env, node):
    """Try to evaluate a type printing the result or reporting errors.

    Args:
        env: The type environment in which to evaluate the expression.
        node: The root node of the abstract syntax tree of the expression.

    Returns:
        None
    """
    print(str(node) + " : ", end=' ')
    try:
        t = analyse(node, env)
        print(str(t))
    except (ParseError, InferenceError) as e:
        print(e)


def main():
    """The main example program.

    Sets up some predefined types using the type constructors TypeVariable,
    TypeOperator and Function.  Creates a list of example expressions to be
    evaluated. Evaluates the expressions, printing the type or errors arising
    from each.

    Returns:
        None
    """

    var1 = TypeVariable()
    var2 = TypeVariable()
    pair_type = TypeOperator("*", (var1, var2))

    var3 = TypeVariable()

    my_env = {"pair": Function(var1, Function(var2, pair_type)),
              "true": Bool,
              "cond": Function(Bool, Function(var3, Function(var3, var3))),
              "zero": Function(Integer, Bool),
              "pred": Function(Integer, Integer),
              "times": Function(Integer, Function(Integer, Integer))}

    pair = Apply(Apply(Ident("pair"), Apply(Ident("f"), Ident("4"))), Apply(Ident("f"), Ident("true")))

    examples = [
        # factorial
        Letrec("factorial",  # letrec factorial =
               Lambda("n",  # fn n =>
                      Apply(
                          Apply(  # cond (zero n) 1
                              Apply(Ident("cond"),  # cond (zero n)
                                    Apply(Ident("zero"), Ident("n"))),
                              Ident("1")),
                          Apply(  # times n
                              Apply(Ident("times"), Ident("n")),
                              Apply(Ident("factorial"),
                                    Apply(Ident("pred"), Ident("n")))
                          )
                      )
                      ),  # in
               Apply(Ident("factorial"), Ident("5"))
               ),

        # Should fail:
        # fn x => (pair(x(3) (x(true)))
        Lambda("x",
               Apply(
                   Apply(Ident("pair"),
                         Apply(Ident("x"), Ident("3"))),
                   Apply(Ident("x"), Ident("true")))),

        # pair(f(3), f(true))
        Apply(
            Apply(Ident("pair"), Apply(Ident("f"), Ident("4"))),
            Apply(Ident("f"), Ident("true"))),

        # let f = (fn x => x) in ((pair (f 4)) (f true))
        Let("f", Lambda("x", Ident("x")), pair),

        # fn f => f f (fail)
        Lambda("f", Apply(Ident("f"), Ident("f"))),

        # let g = fn f => 5 in g g
        Let("g",
            Lambda("f", Ident("5")),
            Apply(Ident("g"), Ident("g"))),

        # example that demonstrates generic and non-generic variables:
        # fn g => let f = fn x => g in pair (f 3, f true)
        Lambda("g",
               Let("f",
                   Lambda("x", Ident("g")),
                   Apply(
                       Apply(Ident("pair"),
                             Apply(Ident("f"), Ident("3"))
                             ),
                       Apply(Ident("f"), Ident("true"))))),

        # Function composition
        # fn f (fn g (fn arg (f g arg)))
        Lambda("f", Lambda("g", Lambda("arg", Apply(Ident("g"), Apply(Ident("f"), Ident("arg"))))))
    ]

    for example in examples:
        tryExp(my_env, example)


if __name__ == '__main__':
    main()
