# Trabajo practico 2 - Ejercicio 1
# a) Implement decision tree with Shannon entropy
# b) Add additional training example and reconstruct decision tree

from graphviz import Digraph

if __name__ == '__main__':
    ggg = Digraph(comment="Testing out")
    ggg.node('A')
    ggg.node('B')
    ggg.node('C')
    ggg.edge('A','B')
    ggg.edge('A','C',constraint='false')
    #ggg.render("./out/ASD.gv")
    ggg.view()