# Bayesian Network
import numpy as np
import pandas as pd
import click
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_FILEPATH_DEFAULT = f"{dir_path}/data/binary.csv"
TABLE = {}
AS = range(0,2)
GRES = range(0,2)
GPAS = range(0,2)
RS = range(1,5)


def p_a_with_gre_gpa_r(dataset, a, r, gre, gpa):
    text = f"P(A={a}|GRE={gre},GPA={gpa},R={r})"
    if text not in TABLE:
        TABLE[text] = dataset.get((a,gre,gpa,r)) / dataset.get([(a_,gre,gpa,r) for a_ in AS]).sum()
    return TABLE[text]


def p_gre_with_r(dataset, r, gre):
    text = f"P(GRE={gre}|R={r})"
    if text not in TABLE:
        TABLE[text] = dataset.get([(a,gre,gpa,r) for a in AS for gpa in GPAS]).sum() / \
                      dataset.get([(a,gre_,gpa,r) for a in AS for gpa in GPAS for gre_ in GRES]).sum()
    return TABLE[text]


def pp_gpa_with_r(dataset,r,gpa):
    text = f"P(GPA={gpa}|R={r})"
    if text not in TABLE:
        TABLE[text] = dataset.get([(a,gre,gpa,r) for a in AS for gre in GRES]).sum() / \
                      dataset.get([(a,gre,gpa_,r) for a in AS for gre in GRES for gpa_ in GPAS]).sum()
    return TABLE[text]


def pp_r(dataset,r):
    text = f"P(R={r})"
    if text not in TABLE:
        TABLE[text] = dataset.get([(a,gre,gpa,r) for a in AS for gre in GRES for gpa in GPAS]).sum() / dataset.sum()
    return TABLE[text]


def pp_a_gre_gpa_r(dataset,a,r,gre,gpa):
    #  P(A = a, R = r, GRE = gre, GPA = gpa) = P(A = a/F_admit) * P(rank = r) * P(GRE = gre/F_GRE) * P(GPA = gpa/F_GPA)
    #  P(A = a, R = r, GRE = gre, GPA = gpa) = P(A = a/GRE=gre,GPA=gpa,R=r) *
    #                                        P(rank = r) *
    #                                        P(GRE = gre/R=r) *
    #                                        P(GPA = gpa/R=r)
    text = f"P(A={a},GRE={gre},GPA={gpa},R={r})"
    if text not in TABLE:
        TABLE[text] = p_a_with_gre_gpa_r(dataset, a=a, r=r, gre=gre, gpa=gpa) * \
                      pp_r(dataset, r) * \
                      p_gre_with_r(dataset, r=r, gre=gre) * \
                      pp_gpa_with_r(dataset,r=r,gpa=gpa)
    return TABLE[text]


def pp_a_with_r(dataset,a,r):
    text = f"P(A={a}|R={r})"
    # P(A=a|R=r) = P(A=a,R=r) / P(R=r)
    # P(A=a|R=r) = SumGRE SumGPA P(A=a,R=r,gpe,gpa) / P(R=r)
    if text not in TABLE:
        TABLE[text] = np.array([pp_a_gre_gpa_r(dataset, a=a,r=r, gre=gre, gpa=gpa) for gre in GRES for gpa in GPAS]).sum() / \
            pp_r(dataset,r=r)
    return TABLE[text]


def complete_table(dataset):

    [p_a_with_gre_gpa_r(dataset, a=a, gre=gre, gpa=gpa, r=r)
     for a in AS
     for r in RS
     for gre in GRES
     for gpa in GPAS]

    [p_gre_with_r(dataset, gre=gre, r=r)
     for r in RS
     for gre in GRES]

    [pp_gpa_with_r(dataset,gpa=gpa,r=r)
           for r in RS
           for gpa in GPAS]

    [pp_r(dataset,r=r) for r in RS]

    [pp_a_with_r(dataset, a=a, r=r)
            for a in AS
            for r in RS]


def get_dataset(data_filepath):
    # Open file
    dataset = pd.read_csv(data_filepath)
    # map to boolean
    dataset["gre"] = dataset.apply(lambda x: 1 if x["gre"] >= 500 else 0,axis=1)
    dataset["admit"] = dataset.apply(lambda x: 1 if x["admit"] == 1 else 0,axis=1)
    dataset["gpa"] = dataset.apply(lambda x: 1 if x["gpa"] >= 3 else 0,axis=1)
    return dataset


def ex_a(a=0, r=1):
    key = f'P(A={a}|R={r})'
    print(f"Ex a => {key} = {TABLE[key]}")


def ex_b(a=1,r=2,gre=0,gpa=1):
    key = f'P(A={a}|GRE={gre},GPA={gpa},R={r})'
    print(f"Ex b => {key} = {TABLE[key]}")

@click.command(name="e1_4")
@click.option("--data-filepath", default=DATA_FILEPATH_DEFAULT)
def main(data_filepath):
    dataset = get_dataset(data_filepath)
    dataset = dataset.append(
        [{"admit": a, "gre": gre, "gpa": gpa, "rank": r} for a in AS for gre in GRES for gpa in GPAS for r in RS])
    dataset = dataset.groupby(dataset.columns.tolist(),as_index=False).size()
    # TO apply Laplace correction comment next line
    # dataset[1] = dataset[1] - 1
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

    complete_table(dataset)
    ex_a()
    ex_b()


if __name__ == '__main__':
    main()