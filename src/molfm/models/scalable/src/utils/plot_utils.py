# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot(sns_style="whitegrid"):
    params = {"text.usetex": False, "mathtext.fontset": "stixsans"}
    plt.rcParams.update(params)
    sns.set_style(sns_style)
    sns.set_context("paper", font_scale=2)
    plt.rc("font", family="serif")
