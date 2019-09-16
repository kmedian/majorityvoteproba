import numpy as np


def majority_vote_proba(x):
    # count how many proba x are greater than 0.5
    cnt = np.sum(x >= 0.5, axis=1).astype(np.uint16)

    # the vote per example
    vote = cnt > int(x.shape[1] / 2)

    # initialize proba y
    y = np.ones(shape=vote.shape, dtype=np.float16) * .5

    # set x<0.5 to zero and add sum(x-0.5)/n
    xp = (x - .5).astype(np.float16)
    xp[xp < 0] = 0  # set negative to zero
    xp[~vote, :] = 0  # set all vote=false examples to 0
    y += xp.mean(axis=1)  # add to 0.5
    del xp

    # set x>=0.5 to zero and add sum(x-0.5)/n
    xn = (x - .5).astype(np.float16)
    xn[xn >= 0] = 0  # set positive to zero
    xn[vote, :] = 0  # set all vote=true examples to 0
    y += xn.mean(axis=1)  # add to 0.5
    del xn

    # done
    return y, vote.astype(np.uint8), cnt
