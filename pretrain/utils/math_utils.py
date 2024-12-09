# -*- coding: utf-8 -*-
"""
@description: 
"""


def edit_distance_word(word, char_set):
    """
    all edits that are one edit away from 'word'
    :param word:
    :param char_set:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
    return set(transposes + replaces)


def get_sub_array(nums):
    """
    取所有连续子串，
    [0, 1, 2, 5, 7, 8]
    => [[0, 3], 5, [7, 9]]
    :param nums: sorted(list)
    :return:
    """
    ret = []
    ii = 0
    for i, c in enumerate(nums):
        if i == 0:
            pass
        elif i <= ii:
            continue
        elif i == len(nums) - 1:
            ret.append([c])
            break
        ii = i
        cc = c
        # get continuity Substring
        while ii < len(nums) - 1 and nums[ii + 1] == cc + 1:
            ii = ii + 1
            cc = cc + 1
        if ii > i:
            ret.append([c, nums[ii] + 1])
        else:
            ret.append([c])
    return ret


def find_all_idx2(lst, item):
    """
    取列表中指定元素的所有下标
    :param lst: 列表或字符串
    :param item: 指定元素
    :return: 下标列表
    """
    ids = []
    for i in range(len(lst)):
        if item == lst[i]:
            ids.append(i)
    return ids


def find_all_idx(lst, item):
    """
    取列表中指定元素的所有下标
    :param lst: 列表或字符串
    :param item: 指定元素
    :return: 下标列表
    """
    ids = []
    pos = -1
    for i in range(lst.count(item)):
        pos = lst.index(item, pos + 1)
        if pos > -1:
            ids.append(pos)
    return ids


def edit_distance_dp(str1: str, str2: str) -> int:
    """
    计算两个字符串的编辑距离
    Args:
        str1:
        str2:

    Returns:
        int: 编辑距离
    """
    if not str1:
        return len(str2)
    if not str2:
        return len(str1)

    dp = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]

    for i in range(0, len(str1) + 1):
        dp[i][0] = i

    for j in range(0, len(str2) + 1):
        dp[0][j] = j

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[-1][-1]


def edit_distance(str1, str2):
    try:
        # very fast
        # http://stackoverflow.com/questions/14260126/how-python-levenshtein-ratio-is-computed
        import Levenshtein
        d = Levenshtein.distance(str1, str2) / float(max(len(str1), len(str2)))
    except:
        # https://docs.python.org/2/library/difflib.html
        import difflib
        d = 1.0 - difflib.SequenceMatcher(lambda x: x == " ", str1, str2).ratio()
    return d


if __name__ == "__main__":
    l = [1, 2, 3, 4, 2, 3, 4]
    item = 2
    print(find_all_idx(l, item))

    l = '我爱中国，我是中国人'
    item = '中国'
    print(find_all_idx(l, item))
