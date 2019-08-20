#author: xiaoyang
import os
import shutil
import traceback
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-src",required = True,
    help = "source path")
ap.add_argument("-tar",required = True,
    help = "target path")
args = vars(ap.parse_args())

def move_file(src_path, dst_path, file):
    print ('from : ', src_path)
    print ('to : ', dst_path)
    try:
        # cmd = 'chmod -R +x ' + src_path
        # os.popen(cmd)
        f_src = os.path.join(src_path, file)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, file)
        #shutil.move(f_src, f_dst)
        shutil.copy(f_src, f_dst)
    except Exception as e:
        print
        'move_file ERROR: ', e
        traceback.print_exc()

if __name__ == '__main__':
    src_files_list = [f for f in os.listdir(args['src']) if f.endswith('.jpg')]
    for file in src_files_list:
        move_file(args['src'],args['tar'],file)


