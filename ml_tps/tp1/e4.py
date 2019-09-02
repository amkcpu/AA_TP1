# Bayesian Network
import numpy as np
import pandas as pd
import click
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/binary.csv"

def get_dataset(data_filepath):
    # Open file
    dataset = pd.read_csv(data_filepath)
    # map to boolean
    dataset["gre"] = dataset.apply(lambda x: 1 if x["gre"] >= 500 else 0,axis=1)
    dataset["admit"] = dataset.apply(lambda x: 1 if x["admit"] == 1 else 0,axis=1)
    dataset["gpa"] = dataset.apply(lambda x: 1 if x["gpa"] >= 3 else 0,axis=1)
    return dataset


def p_a_with_r_gre_gpa(dataset,a,r,gre,gpa,tabs=0):
    print(f"{t_text(tabs)}Obtaining P(A={a}|GRE={gre},GPA={gpa},R={r})")
    ans = p_with_general(dataset,aas=[a], rs=[r], gres=[gre], gpas= [gpa], tabs = tabs+1)
    print(f"{t_text(tabs)}P(A={a}|GRE={gre},GPA={gpa},R={r}) = {ans} from dataset")
    return ans


def t_text(tabs):
    return '\t'*tabs

def pp_general(dataset, a, r, gre, gpa, tabs):
    # P(A=a,GRE=gre,GPA=gpa,R=r) = P(A=a|Father(A)) * P(R=r|Father(R)) * P(GRE=gre|Father(GRE)) * P(GPA=gpa|Father(GPA))
    # P(A=a,GRE=gre,GPA=gpa,R=r) = P(A=a|GRE=gre,GPA=gpa,R=r) * P(R=r) * P(GRE=gre|R=r) * P(GPA=gpa|R=r)
    print(f"{t_text(tabs=tabs)}Obtaining P(A={a},GRE={gre}, GPA={gpa}, R={r})")
    a_with_r_gre_gpa = p_a_with_r_gre_gpa(dataset, a=a, r=r, gre=gre, gpa=gpa, tabs=tabs+1)
    r_ = p_r(dataset, r=r,tabs=tabs+1)
    gre_with_r = p_gre_with_r(dataset, r=r, gre=gre,tabs=tabs+1)
    gpa_with_r = p_gpa_with_r(dataset, r=r, gpa=gpa, tabs=tabs+1)
    ans = a_with_r_gre_gpa * r_ * gre_with_r * gpa_with_r
    print(f"{t_text(tabs=tabs)}P(A={a},GRE={gre}, GPA={gpa}, R={r}) = {ans}")
    return ans


def p_with_general(dataset, aas=[0, 1], rs=[1,2,3,4], gres=[0, 1], gpas=[0, 1], tabs=0):
    tabs = tabs+1
    print(f"{t_text(tabs=tabs)}Obtaining P(A={aas}|GRE={gres}, GPA={gpas}, R={rs})")
    ans = dataset.get([(a,gre,gpa,r) for a in aas for r in rs for gre in gres for gpa in gpas]).sum() / dataset.sum()
    print(f"{t_text(tabs)}P(A={aas}|GRE={gres}, GPA={gpas}, R={rs}) from dataset = {ans}")
    return ans


def p_general(dataset, tabs, aas=[0, 1], rs=[1,2,3,4], gres=[0, 1], gpas=[0, 1]):
    print(f"{t_text(tabs=tabs)}Obtaining P(A={aas},GRE={gres}, GPA={gpas}, R={rs})")
    ans = np.array([pp_general(dataset, a=a, gre=gre,gpa=gpa,r=r, tabs=tabs+1) for a in aas for r in rs for gre in gres for gpa in gpas]).sum()
    print(f"{t_text(tabs)}P(A={aas},GRE={gres}, GPA={gpas}, R={rs}) = {ans}")
    return ans


def p_a_r_gre_gpa(dataset,a,r,gre,gpa,tabs):
    tabs +=1
    # P(A=a,GRE=gre, GPA=gpa, R=r) with a,gre,gpa in {0,1}, r in 1..4 and P(a,gre,gpa,r) from dataset
    print(f"{t_text(tabs=tabs)}Obtaining P(A={a},GRE={gre},GPA={gpa},R={r})")
    ans = p_general(dataset,aas=[a],rs=[r],gres=[gre],gpas=[gpa],tabs=tabs)
    print(f"{t_text(tabs=tabs)}Obtaining P(A={a},GRE={gre},GPA={gpa}|R={r})")
    return ans



def p_r_gre(dataset,r,gre,tabs):
    tabs +=1
    # P(GRE=gre,R=r) with gre in {0,1}, r in {1,2,3,4} and P(g,r) from dataset
    print(f"{t_text(tabs=tabs)}Obtaining P(GRE={gre},R={r})")
    ans = p_general(dataset, gres=[gre], rs=[r], tabs=tabs)
    print(f"{t_text(tabs=tabs)}P(GRE={gre},R={r}) = {ans}")
    return ans


def p_r(dataset,r, tabs=0):
    # P(R=r) from dataset
    print(f"{t_text(tabs=tabs)}Obtaining P(R={r})")
    ans = p_with_general(dataset,rs=[r],tabs=tabs+1)
    print(f"{t_text(tabs)}P(R={r}) from dataset = {ans}")
    return ans


def p_gre_with_r(dataset,r,gre,tabs):
    # P(GRE=g|R=r) from table
    print(f"{t_text(tabs=tabs)}Obtaining P(GRE={gre}|R={r})")
    ans = p_with_general(dataset,gres=[gre],rs=[r],tabs=tabs)
    print(f"{t_text(tabs)}P(GRE={gre}|R={r}) = {ans} from dataset")
    return ans


def p_r_gpa(dataset,r, gpa, tabs):
    tabs += 1
    print(f"{t_text(tabs=tabs)}Obtaining P(GPA={gpa},R={r})")
    # P(GPA=gpa,R=r) with gpa in {0,1}, r in {1,2,3,4} and P(gpa,r) from dataset
    return p_general(dataset, gpas=[gpa], rs=[r],tabs=tabs)


def p_gpa_with_r(dataset,r,gpa, tabs):
    # P(GRE=g|R=r) from table
    print(f"{t_text(tabs=tabs)}Obtaining P(GPA={gpa}|R={r})")
    ans = p_with_general(dataset, gpas=[gpa], rs=[r], tabs= tabs+1)
    print(f"{t_text(tabs)}P(GPA={gpa}|R={r}) = {ans} from dataset")
    return ans


def p_a_with_r(dataset,a,r,tabs=0):
    print(f"{t_text(tabs=tabs)}Obtaining P(A={a}|R={r})")
    # P(A=a,R=r) = P(A=a|GRE,GPA,R=r) * P(R=r)
    # P(A=a|R=r) = P(A=a,R=r) / P(R=r)
    a_r = p_a_r(dataset, a=a, r=r, tabs=tabs+1)
    r_ = p_r(dataset,r=r, tabs=tabs+1)
    ans = a_r / r_
    #print("P(A=a,R=r) = P(A=a|GRE,GPA,R=r) * P(R=r)")
    print(f"{t_text(tabs)}P(A={a}|R={r}) = P(A={a},R={r}) / P(R={r}) = {a_r} / {r_}) = {ans}")
    return ans


def p_a_r(dataset,a,r,tabs):
    return p_general(dataset,aas=[a], rs=[r], tabs=tabs)

def ex_a(dataset):
    # P(a=0|R=1) = P(a=0,R=1) / P(R=1)
    ans = p_a_with_r(dataset,a=0,r=1,tabs=0)
    print(ans)
    return ans


def ex_b(dataset):
    # 4.b P(a=1|R=2,gre=0,gpa=1)
    ans = p_a_with_r_gre_gpa(dataset,a=1,r=2,gre=0,gpa=1)
    print(ans)
    return ans

@click.command(name="e1_4")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
def main(data_filepath):
    dataset = get_dataset(data_filepath)
    # Laplace correction
    dataset = dataset.append(
        [{"admit": a, "gre": gre, "gpa": gpa, "rank": r} for a in [0, 1] for gre in [0, 1] for gpa in [0, 1] for r in
         range(1, 5)])
    dataset = dataset.groupby(dataset.columns.tolist(),as_index=False).size()
    # The graph is      rank
    #           gre <----┘|└----> gpa
    #            |        v        |
    #            └----->admit<-----┘
    # So we have: P(R=r) with r in {1,2,3,4}
    #             P(GRE=g) with g in {0,1} and P(g) = Sum (r in 1..4) P(g|r) * P(r)
    #             P(GRE=g,R=r) with g in {0,1}, r in {1,2,3,4} and P(g,r) = P(g|r) * P(r)
    #             P(GPA=g) with g in {0,1} and P(g) = Sum (r in 1..4) P(g|r) * P(r)
    #             P(GPA=g,R=r) with g in {0,1}, r in {1,2,3,4} and P(g,r) = P(g|r) * P(r)
    #             P(A=a) with a in {0,1} and P(a) = Sum (r in 1..4) Sum(gre in 0..1) Sum(gpa in 0..1) P(a|r,gre,gpa) *
    #                                               P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,GPA=gpa) with a,gpa in {0,1} and P(a,gpa) =
    #               Sum (r in 1..4) Sum(gre in 0..1) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,GRE=gre) with a,gre in {0,1} and P(a,gre) =
    #               Sum (r in 1..4) Sum(gpa in 0..1) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,R=r) with a in {0,1}, r in 1..4 and P(a,r) =
    #               Sum(gre in 0..1) Sum(gpa in 0..1) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,GPA=gpa, R=r) with a,gpa in {0,1}, r in 1..4 and P(a,gpa,r) =
    #               Sum(gre in 0..1) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,GRE=gre, R=r) with a,gre in {0,1}, r in 1..4 and P(a,gre,r) =
    #               Sum(gpa in 0..1) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,GRE=gre, GPA=gpa) with a,gre,gpa in {0,1} and P(a,gre,gpa) =
    #               Sum(r in 1..4) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    #             P(A=a,GRE=gre, GPA=gpa, R=r) with a,gre,gpa in {0,1}, r in 1..4 and P(a,gre,gpa,r) from dataset
    #             P(A=a|R=r) = P(a,r) / P(r)
    #             P(A=a|GRE=gre) = P(a,gre) / P(gre)
    #             P(A=a|GPA=gpa) = P(a,gpa) / P(gpa)
    #             P(A=a|GRE=gre,R=r) = P(a,gre,r) / P(gre,r)
    #             P(A=a|GPA=gpa,R=r) = P(a,gpa,r) / P(gpa,r)
    #             P(A=a|GRE=gre,GPA=gpa) = P(a,gpa,gre) / P(gre,gpa) = P(a,gpa,gre) / (P(gre)*P(gpa))
    #             P(A=a|GRE=gre,GPA=gpa,R=r) = P(a,gre,gpa,r) / P(gra,gpe,r) = P(a,gre,gpa,r) / (P(gra,r) *P(gpe,r))



    # 4.a P(a=0|R=1) = P(a=0,R=1) / P(R=1)
    # P(a=0|R=1) = P(a=0,R=1) / P(R=1) =
    # = sum(gpa{0,1}) sum(gre{0,1}) P(a|r,gre,gpa) * P(gre|r) * P(gpa|r) * P(r)
    # = sum(gpa{0,1}) sum(gre{0,1}) (P(a,gpa,gpe,r) / P(r,gpa,gre) = P(a,gpa,gpe,r) / (P(r,gpa)*P(r,gre))) *
    # P(gre|r) * P(gpa|r) * P(r)
    #p_a_with_r_gre_gpa(dataset,0,1,0,0)
    #p_a_with_r_gre_gpa(dataset,1,1,0,0)


    ex_a(dataset)
    o = 5


if __name__ == '__main__':
    main()