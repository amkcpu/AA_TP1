# Bayesian Network
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


def p_a_with_r_gre_gpa(dataset,a,r,gre,gpa):
    # P(A=a|GRE=gre,GPA=gpa,R=r) = P(a,gre,gpa,r) / P(gra,gpe,r) = P(a,gre,gpa,r) / (P(gre,r) *P(gpa,r))
    a_r_gre_gpa = p_a_r_gre_gpa(dataset, a=a, r=r, gre=gre, gpa=gpa)
    r_gre_gpa = p_r_gre_gpa(dataset,r=r,gre=gre,gpa=gpa)
    ans = a_r_gre_gpa / r_gre_gpa
    print(f"P(A={a}|GRE={gre},GPA={gpa},R={r}) = P(A={a},GRE={gre},GPA={gpa},R={r}) / P(GPA={gpa},GPE={gre},R={r}) = "
          f"{a_r_gre_gpa} / {r_gre_gpa} = {ans}")
    return ans


def p_general(dataset, aas=None, rs=None, gres=None, gpas=None):
    if gpas is None:
        gpas = [0, 1]
    if gres is None:
        gres = [0, 1]
    if rs is None:
        rs = [1, 2, 3, 4]
    if aas is None:
        aas = [0, 1]
    ans = dataset.get([(a,gre,gpa,r) for a in aas for r in rs for gre in gres for gpa in gpas]).sum() / dataset.sum()
    print(f"P(A={aas},GRE={gres}, GPA={gpas}, R={rs}) from dataset = {ans}")
    return ans


def p_a_r_gre_gpa(dataset,a,r,gre,gpa):
    # P(A=a,GRE=gre, GPA=gpa, R=r) with a,gre,gpa in {0,1}, r in 1..4 and P(a,gre,gpa,r) from dataset
    return p_general(dataset,aas=[a],rs=[r],gres=[gre],gpas=[gpa])


def p_r_gre_gpa(dataset,r,gre,gpa):
    # P(gra, gpe, r) = P(gre, r) * P(gpa, r)
    gre_with_r = p_gre_with_r(dataset, r=r, gre=gre)
    gpa_with_r = p_gpa_with_r(dataset, r=r, gpa=gpa)
    r_ = p_r(dataset, r=r)
    ans = gre_with_r * gpa_with_r * r_
    print(f"P(GRE={gre}, GPA={gpa}, R={r}) = P(GRE={gre}| R={r}) * P(GPA={gpa}| R={r}) * P(R={r}) = {gre_with_r} * "
          f"{gpa_with_r} * {r_} = {ans}")
    return ans


def p_r_gre(dataset,r,gre):
    # P(GRE=gre,R=r) with g in {0,1}, r in {1,2,3,4} and P(g,r) = P(g|r) * P(r)
    gre_with_r = p_gre_with_r(dataset, r=r, gre=gre)
    r_ = p_r(dataset,r=r)
    ans = gre_with_r * r_
    print(f"P(GRE={gre},R={r}) = P(GRE={gre}|R={r}) * P(R={r}) = {gre_with_r} * {r_}= {ans}")
    return ans


def p_r(dataset,r):
    # P(R=r) from dataset
    ans = dataset.get([(a,gre,gpa,r) for a in [0,1] for gre in [0,1] for gpa in [0,1]]).sum() / dataset.sum()
    print(f"P(R={r}) from dataset = {ans}")
    return ans


def p_gre_with_r(dataset,r,gre):
    # P(GRE=g|R=r) from dataset
    ans = dataset.get([(a,gre,gpa,r) for a in [0,1] for gpa in [0,1]]).sum() / dataset.sum()
    print(f"P(GRE={gre}|R={r}) from dataset = {ans}")
    return ans


def p_r_gpa(dataset,r, gpa):
    # P(GPA=gpa,R=r) with gpa in {0,1}, r in {1,2,3,4} and P(gpa,r) = P(gpa|r) * P(r)
    gpa_with_r = p_gpa_with_r(dataset, gpa=gpa, r=r)
    r_ = p_r(dataset,r=r)
    ans = gpa_with_r * r_
    print(f"P(GPA={gpa},R={r}) = P(GRE={gpa}|R={r}) * P(R={r}) = {gpa_with_r} * {r_} = {ans}")
    return ans


def p_gpa_with_r(dataset,r,gpa):
    # P(GPA=g|R=r) from dataset
    ans = dataset.get([(a,gre,gpa,r) for a in [0,1] for gre in [0,1]]).sum() / dataset.get([(a,gre,gpa_,r) for a in [0,1] for gre in [0,1] for gpa_ in [0,1]]).sum()
    print(f"P(GPA={gpa}|R={r}) from dataset = {ans}")
    return ans


def ex_a(dataset):
    # P(a=0|R=1) = P(a=0,R=1) / P(R=1)
    pass


def ex_b(dataset):
    # 4.b P(a=1|R=2,gre=0,gpa=1) = P(a=1,R=2,gre=0,gpa=1) / P(R=2,gre=1,gpa=1)
    ans = p_a_with_r_gre_gpa(dataset,1,2,0,1)
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
    p_a_with_r_gre_gpa(dataset,0,1,0,0)
    p_a_with_r_gre_gpa(dataset,1,1,0,0)


    ex_b(dataset)
    o = 5


if __name__ == '__main__':
    main()