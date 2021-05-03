# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os

# Generate the json(one article one line) from the uncompressed files.

if __name__ == "__main__":

    dir = ['data_0','data_1','data_2','data_3']
    cur_path = '/mnt/GPT2Data'
    json_name = "merged_files.json"
    output_path = os.path.join(cur_path, json_name)
    text = {}
    cnt = 0
    with open(output_path, 'w') as outputfile:
        for dir_name in dir:
            dir_path = os.path.join(cur_path, dir_name)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                url = file_name[:-4].replace('-','')
                with open(file_path, 'r') as inputfile:
                    cnt += 1
                    if cnt % 1024 == 0:
                        print("Merging at ", cnt, flush=True)
                    text['text'] = inputfile.read()
                    text['url'] = url
                    outputfile.write(json.dumps(text) + '\n')
    print("Merged file", flush=True)
    