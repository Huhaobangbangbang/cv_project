#!/usr/bin/env python

from __future__ import print_function
import argparse
import os.path as osp
import sys
import time
import pandas as pd


def print_bar(title='', width=80):
    if title:
        title = ' ' + title + ' '
    length = len(title)
    if length % 2 == 0:
        length_left = length_right = length // 2
    else:
        length_left = length // 2
        length_right = length - length_left
    print('=' * (width // 2 - 1 - length_left) +
          title + '=' * (width // 2 - length_right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('-1', '--once', action='store_true')
    args = parser.parse_args()

    log_file = args.log_file

    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    while True:
        try:
            ext = osp.splitext(log_file)[-1]
            if ext == '.json':
                df = pd.read_json(log_file)
            elif ext == '.csv':
                df = pd.read_csv(log_file)
            else:
                print('Unsupported file extension: {}'.format(log_file))
                sys.exit(1)
            df = df.set_index(['epoch', 'iteration'])

            train_cols, valid_cols, else_cols = [], [], []
            for col in df.columns:
                if col.startswith('validation/') or col.startswith('valid/'):
                    valid_cols.append(col)
                elif col.startswith('train/'):
                    train_cols.append(col)
                else:
                    else_cols.append(col)

            if args.all:
                print(df.to_string())
                break

            width = len('  '.join(['epoch', 'iteration']) + '  '.join(train_cols + else_cols) + '   ')

            print(chr(27) + "[2J")

            log_dir = osp.dirname(log_file)
            param_files = [osp.join(log_dir, f) for f in ['params.yaml', 'config.yaml']]
            exists = [osp.exists(f) for f in param_files]
            if any(exists):
                import yaml
                param_file = param_files[exists.index(True)]
                data = yaml.load(open(param_file))
                print_bar('params', width=width)
                print(yaml.safe_dump(data, default_flow_style=False))
                print_bar('', width=width)

            print('log_file: %s' % log_file)

            if df.empty:
                time.sleep(1)
                continue

            try:
                df_train = df[train_cols + else_cols].dropna(thresh=len(train_cols))
            except:
                df_train = df[train_cols + else_cols].dropna()
            if not df_train.empty:
                print_bar('train', width=width)
                print(df_train.tail(n=5))
                print()

            try:
                df_valid = df[valid_cols + else_cols].dropna(thresh=len(valid_cols))
            except:
                df_valid = df[valid_cols + else_cols].dropna()
            if not df_valid.empty:
                print_bar('valid', width=width)
                print(df_valid.tail(n=3))
                print()

                for col in valid_cols:
                    if 'loss' in col:
                        print_bar('min:%s' % col, width=width)
                        idx = df[col].idxmin()
                    else:
                        print_bar('max:%s' % col, width=width)
                        idx = df[col].idxmax()
                    try:
                        print(df.ix[idx][valid_cols + else_cols].dropna(thresh=len(valid_cols)))
                    except:
                        print(df.ix[idx][valid_cols + else_cols].dropna())

            print_bar(width=width)

            if args.once:
                break
            time.sleep(1)
        except KeyboardInterrupt:
            break
        except IOError as e:
            print(chr(27) + "[2J")
            print(e)
            time.sleep(1)
            continue
        except Exception as e:
            print(e)
            break


if __name__ == '__main__':
    main()
