import pandas as pd
from strat.SE.back_pr.back_plot_strat.highlow_cloudlb import plot_check

if __name__ == "__main__":

    back_df = pd.read_excel("back_pr.xlsx", index_col=0)
    print("back_pr.xlsx loaded !")

    row1, row2 = 1041, 1071
    plot_check(back_df, row1 - 2, row2 - 2, prev_plotsize=100)   # i,j <-- excel index - 2
