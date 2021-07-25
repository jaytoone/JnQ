import pandas as pd
from basic_v1.back_pr.back_plot_check.highlow_cloudlb import plot_check

if __name__ == "__main__":

    back_df = pd.read_excel("back_pr.xlsx", index_col=0)
    print("back_pr.xlsx loaded !")

    plot_check(back_df, 706, 724)   # i,j <-- excel index - 2
