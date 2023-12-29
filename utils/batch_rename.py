import os
import tqdm

"""
    批量重命名
"""

if __name__ == '__main__':
    dir_path = r"D:\Study\AiShell_test\output"  # 操作目录
    old_part = "_generated"  # 待替换的部分
    new_part = ""  # 替换为的部分
    
    file_list = os.listdir(dir_path)

    for f in tqdm.tqdm(file_list):
        new_fname = f.replace(old_part, new_part)
        path = os.path.join(dir_path, f)

        new_path = os.path.join(dir_path, new_fname)
        os.rename(path, new_path)

    print("process over")