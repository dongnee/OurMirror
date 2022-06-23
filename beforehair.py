# 헤어스타일 4가지와 이미지 경로를 내보내는 함수
# input : 성별(male, female), 얼굴형(heart, oblong, round, square, oval), 컷/펌(cut, perm), 길이(1-장, 2-중, 3-단, 4-숏, 5-남자)-숫자는 문자형으로 ex) "1"
# output :path 이미지 저장 경로 딕셔너리 형태,  hair 딕셔너리 형태

path = {}
hair = {}

def hairstyle_recom(gender, face_shape, style, hair_length):
    from tensorflow.keras.models import load_model
    from tensorflow.python.keras.utils import np_utils
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Activation
    import tensorflow as tf
    import numpy as np
    import sys
    import os
    from numpy import argmax
    import matplotlib.pyplot as plt             
    import cv2                                 
    from tqdm import tqdm
    import shutil
    from pathlib import Path
    import json
    from collections import OrderedDict
    from tensorflow.keras import models
    from glob import glob
    import matplotlib.image as image

    import PIL
    import torch
    import efficientnet_pytorch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    global path, hair
        
    if gender == "woman":
        if face_shape == "하트형":
            if style == "cut":
                if hair_length == "1":
                    hair = {"1": "뱅", "2": "레이어드컷", "3": "플리츠컷",
                            "4": "보브"}  # female, heart, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/cut/1/bang.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/cut/1/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/cut/1/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Heart/cut/1/vov.jpg"}
                elif hair_length == "2":
                    hair = {"1": "뱅", "2": "단발", "3": "플리츠컷",
                            "4": "보브"}  # female, heart, cut, 중발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/cut/2/bang.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/cut/2/onelength.JPG",
                            "3": "./hair_image/recom_hair/woman/Heart/cut/2/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Heart/cut/2/vov.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니컷", "2": "레이어드컷", "3": "플리츠컷",
                            "4": "태슬컷"}  # female, heart, cut, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/cut/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/cut/3/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/cut/3/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Heart/cut/3/tassel.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드컷", "2": "본컷", "3": "레이어드컷", "4": "리프컷"}  # female, heart, cut, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/cut/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/cut/4/born.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/cut/4/layered.JPG",
                            "4": "./hair_image/recom_hair/woman/Heart/cut/4/leaf.jpg"}


            elif style == "perm":
                if hair_length == "1":
                    hair = {"1": "빌드펌", "2": "c컬", "3": "엘리자벳펌",
                            "4": "레이어드펌"}  # female, heart, perm, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/perm/1/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/perm/1/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/perm/1/elisabete.JPG",
                            "4": "./hair_image/recom_hair/woman/Heart/perm/1/layered.jpg"}
                elif hair_length == "2":
                    hair = {"1": "빌드펌", "2": "c컬", "3": "엘리자벳펌",
                            "4": "레이어드펌"}  # female, heart, perm, 중발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/perm/2/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/perm/2/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/perm/2/elisabete.JPG",
                            "4": "./hair_image/recom_hair/woman/Heart/perm/2/layered.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니펌", "2": "브룩펌", "3": "c컬",
                            "4": "웨이브펌"}  # female, heart, perm, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/perm/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/perm/3/brooke.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/perm/3/ccurl.jpg",
                            "4": "./hair_image/recom_hair/woman/Heart/perm/3/wave.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드펌", "2": "코튼펌", "3": "다이앤",
                            "4": "리프펌"}  # female, heart, perm, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Heart/perm/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Heart/perm/4/cotton.jpg",
                            "3": "./hair_image/recom_hair/woman/Heart/perm/4/diann.jpg",
                            "4": "./hair_image/recom_hair/woman/Heart/perm/4/leaf.jpg"}

        elif face_shape == "긴형":
            if style == "cut":
                if hair_length == "1":
                    hair = {"1": "시스루뱅", "2": "허쉬컷", "3": "레이어드컷",
                            "4": "플리츠"}  # female, oblong, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/cut/1/bang.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/cut/1/hush.JPG",
                            "3": "./hair_image/recom_hair/woman/Oblong/cut/1/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Oblong/cut/1/pleats.jpg"}
                elif hair_length == "2":
                    hair = {"1": "시스루뱅", "2": "허쉬컷", "3": "레이어드컷",
                            "4": "플리츠"}  # female, oblong, cut, 중발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/cut/2/bang.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/cut/2/hush.JPG",
                            "3": "./hair_image/recom_hair/woman/Oblong/cut/2/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Oblong/cut/2/pleats.jpg"}
                elif hair_length == "3":
                    hair = {"1": "풀뱅", "2": "레이어드컷", "3": "샤밍컷",
                            "4": "태슬컷"}  # female, oblong, cut, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/cut/3/fullbang.JPG",
                            "2": "./hair_image/recom_hair/woman/Oblong/cut/3/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Oblong/cut/3/shaming.jpg",
                            "4": "./hair_image/recom_hair/woman/Oblong/cut/3/tassel.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드컷", "2": "홀컷", "3": "레이어드컷",
                            "4": "리프컷"}  # female, oblong, cut, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/cut/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/cut/4/hole.JPG",
                            "3": "./hair_image/recom_hair/woman/Oblong/cut/4/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Oblong/cut/4/leaf.jpg"}


            elif style == "perm":
                if hair_length == "1":
                    hair = {"1": "빌드펌", "2": "c컬펌", "3": "엘리자벳펌",
                            "4": "레이어드펌"}  # female, oblong, perm, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/perm/1/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/perm/1/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Oblong/perm/1/elisabete.JPG",
                            "4": "./hair_image/recom_hair/woman/Oblong/perm/1/layered.jpg"}
                elif hair_length == "2":
                    hair = {"1": "빌드펌", "2": "c컬펌", "3": "엘리자벳펌",
                            "4": "레이어드펌"}  # female, oblong, perm, 중발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/perm/2/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/perm/2/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Oblong/perm/2/elisabete.JPG",
                            "4": "./hair_image/recom_hair/woman/Oblong/perm/2/layered.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니펌", "2": "브룩펌", "3": "c컬펌",
                            "4": "웨이브펌"}  # female, oblong, perm, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/perm/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/perm/3/brooke.jpg",
                            "3": "./hair_image/recom_hair/woman/Oblong/perm/3/ccurl.jpg",
                            "4": "./hair_image/recom_hair/woman/Oblong/perm/3/wave.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드펌", "2": "코튼펌", "3": "다이앤",
                            "4": "리프펌"}  # female, oblong, perm, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oblong/perm/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Oblong/perm/4/cotton.jpg",
                            "3": "./hair_image/recom_hair/woman/Oblong/perm/4/diann.jpg",
                            "4": "./hair_image/recom_hair/woman/Oblong/perm/4/leaf.jpg"}

        elif face_shape == "둥근형":
            if style == "cut":
                if hair_length == "1":
                    hair = {"1": "허쉬컷", "2": "레이어드컷", "3": "플리츠컷",
                            "4": "보브컷"}  # female, round, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/cut/1/hush.JPG",
                            "2": "./hair_image/recom_hair/woman/Round/cut/1/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/cut/1/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/cut/1/vov.jpg"}
                elif hair_length == "2":
                    hair = {"1": "허쉬컷", "2": "레이어드컷", "3": "플리츠컷",
                            "4": "보브컷"}  # female, round, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/cut/2/hush.JPG",
                            "2": "./hair_image/recom_hair/woman/Round/cut/2/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/cut/2/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/cut/2/vov.jpg"}
                elif hair_length == "3":
                    hair = {"1": "레이어드컷", "2": "플리츠컷", "3": "샤밍컷",
                            "4": "태슬컷"}  # female, round, cut, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/cut/3/layered.jpg",
                            "2": "./hair_image/recom_hair/woman/Round/cut/3/pleasts.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/cut/3/shaming.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/cut/3/tassel.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드컷", "2": "본컷", "3": "레이어드컷",
                            "4": "리프컷"}  # female, round, cut, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/cut/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Round/cut/4/born.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/cut/4/layered.JPG",
                            "4": "./hair_image/recom_hair/woman/Round/cut/4/leaf.jpg"}


            elif style == "perm":
                if hair_length == "1":
                    hair = {"1": "빌드펌", "2": "c컬펌", "3": "레이어드펌",
                            "4": "웨이브펌"}  # female, round, perm, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/perm/1/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Round/perm/1/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/perm/1/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/perm/1/wave.jpg"}
                elif hair_length == "2":
                    hair = {"1": "빌드펌", "2": "c컬펌", "3": "레이어드펌",
                            "4": "웨이브펌"}  # female, round, perm, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/perm/2/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Round/perm/2/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/perm/2/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/perm/2/wave.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니펌", "2": "브룩펌", "3": "c컬펌",
                            "4": "웨이브펌"}  # female, round, perm, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/perm/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Round/perm/3/brooke.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/perm/3/ccurl.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/perm/3/wave.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드펌", "2": "코튼펌", "3": "다이앤",
                            "4": "리프펌"}  # female, round, perm, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Round/perm/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Round/perm/4/cotton.jpg",
                            "3": "./hair_image/recom_hair/woman/Round/perm/4/diann.jpg",
                            "4": "./hair_image/recom_hair/woman/Round/perm/4/leaf.jpg"}

        elif face_shape == "사각형":
            if style == "cut":
                if hair_length == "1":
                    hair = {"1": "허쉬컷", "2": "레이어드컷", "3": "플리츠컷",
                            "4": "태슬컷"}  # female, square, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/cut/1/hush.JPG",
                            "2": "./hair_image/recom_hair/woman/Square/cut/1/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/cut/1/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Square/cut/1/tassel.JPG"}
                elif hair_length == "2":
                    hair = {"1": "허쉬컷", "2": "레이어드컷", "3": "플리츠컷",
                            "4": "태슬컷"}  # female, square, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/cut/2/hush.JPG",
                            "2": "./hair_image/recom_hair/woman/Square/cut/2/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/cut/2/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Square/cut/2/tassel.JPG"}
                elif hair_length == "3":
                    hair = {"1": "보니컷", "2": "레이어드컷", "3": "플리츠컷", "4": "태슬컷"}  # 여자, square, cut, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/cut/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Square/cut/3/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/cut/3/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Square/cut/3/tassel.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드컷", "2": "본컷", "3": "레이어드컷", "4": "리프컷"}  # 여자, square, cut, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/cut/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Square/cut/4/born.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/cut/4/layered.JPG",
                            "4": "./hair_image/recom_hair/woman/Square/cut/4/leaf.jpg"}


            elif style == "perm":
                if hair_length == "1":
                    hair = {"1": "c컬펌", "2": "히피펌", "3": "레이어드펌", "4": "미스티펌"}  # 여자, square, perm, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/perm/1/ccurl.jpg",
                            "2": "./hair_image/recom_hair/woman/Square/perm/1/hippie.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/perm/1/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Square/perm/1/misty.JPG"}
                elif hair_length == "2":
                    hair = {"1": "빌드펌", "2": "c컬펌", "3": "엘리자벳펌", "4": "히피펌"}  # 여자, square, perm, 중발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/perm/2/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Square/perm/2/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/perm/2/elisabete.JPG",
                            "4": "./hair_image/recom_hair/woman/Square/perm/2/hippie.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니펌", "2": "브룩펌", "3": "c컬펌", "4": "웨이브펌"}  # 여자, square, perm, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/perm/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Square/perm/3/brooke.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/perm/3/ccurl.jpg",
                            "4": "./hair_image/recom_hair/woman/Square/perm/3/wave.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드펌", "2": "코튼펌", "3": "다이앤", "4": "리프펌"}  # 여자, square, perm, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Square/perm/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Square/perm/4/cotton.jpg",
                            "3": "./hair_image/recom_hair/woman/Square/perm/4/diann.jpg",
                            "4": "./hair_image/recom_hair/woman/Square/perm/4/leaf.jpg"}

        else:  # face = oval
            if style == "cut":
                if hair_length == "1":
                    hair = {"1": "허그컷", "2": "허쉬컷", "3": "레이어드컷", "4": "플리츠컷"}  # 여자, oval, cut, 장발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/cut/1/hug.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/cut/1/hush.JPG",
                            "3": "./hair_image/recom_hair/woman/Oval/cut/1/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Oval/cut/1/pleats.jpg"}
                elif hair_length == "2":
                    hair = {"1": "시스루뱅", "2": "허쉬컷", "3": "단발", "4": "플리츠컷"}  # 여자, oval, cut, 중발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/cut/2/bang.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/cut/2/hush.JPG",
                            "3": "./hair_image/recom_hair/woman/Oval/cut/2/onelength.JPG",
                            "4": "./hair_image/recom_hair/woman/Oval/cut/2/pleats.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니컷", "2": "레이어드컷", "3": "플리츠컷", "4": "태슬컷"}  # 여자, oval, cut, 단발 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/cut/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/cut/3/layered.jpg",
                            "3": "./hair_image/recom_hair/woman/Oval/cut/3/pleats.jpg",
                            "4": "./hair_image/recom_hair/woman/Oval/cut/3/tassel.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드컷", "2": "본컷", "3": "레이어드컷", "4": "리프컷"}  # 여자, oval, cut, 숏 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/cut/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/cut/4/born.jpg",
                            "3": "./hair_image/recom_hair/woman/Oval/cut/4/layered.JPG",
                            "4": "./hair_image/recom_hair/woman/Oval/cut/4/leaf.jpg"}


            elif style == "perm":
                if hair_length == "1":
                    hair = {"1": "c컬펌", "2": "히피펌", "3": "레이어드펌", "4": "미스티펌"}  # 여자, oval, perm, 1 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/perm/1/ccurl.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/perm/1/hippie.jpg",
                            "3": "./hair_image/recom_hair/woman/Oval/perm/1/layered.jpg",
                            "4": "./hair_image/recom_hair/woman/Oval/perm/1/misty.JPG"}
                elif hair_length == "2":
                    hair = {"1": "빌드펌", "2": "c컬펌", "3": "엘리자벳펌", "4": "히피펌"}  # 여자, oval, perm, 2 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/perm/2/build.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/perm/2/ccurl.jpg",
                            "3": "./hair_image/recom_hair/woman/Oval/perm/2/elisabete.JPG",
                            "4": "./hair_image/recom_hair/woman/Oval/perm/2/hippie.jpg"}
                elif hair_length == "3":
                    hair = {"1": "보니펌", "2": "브룩펌", "3": "c컬펌", "4": "웨이브펌"}  # 여자, oval, perm, 3 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair./hair_image/recom_hair/woman/Oval/perm/3/bonnie.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/perm/3/brooke.jpg",
                            "3": "./hair_image/recom_hair/woman/Oval/perm/3/ccurl.jpg",
                            "4": "./hair_image/recom_hair/woman/Oval/perm/3/wave.JPG"}
                elif hair_length == "4":
                    hair = {"1": "버드펌", "2": "코튼펌", "3": "다이앤", "4": "리프펌"}  # 여자, oval, perm, 4 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/woman/Oval/perm/4/bird.jpg",
                            "2": "./hair_image/recom_hair/woman/Oval/perm/4/cotton.jpg",
                            "3": "./hair_image/recom_hair/woman/Oval/perm/4/diann.jpg",
                            "4": "./hair_image/recom_hair/woman/Oval/perm/4/leaf.jpg"}



    else:  # 성별 = man
        if face_shape == "하트형":
            if style == "cut":
                if hair_length == "5":  # 남자  헤어 길이 분류는 '남자'만 있음
                    hair = {"1": "댄디컷", "2": "플랫컷", "3": "가일컷", "4": "아이비리그컷"}  # male, heart, cut, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Heart/cut/5/dandy1.jpg",
                            "2": "./hair_image/recom_hair/man/Heart/cut/5/flat1.jpg",
                            "3": "./hair_image/recom_hair/man/Heart/cut/5/gail1.jpg",
                            "4": "./hair_image/recom_hair/man/Heart/cut/5/ivyleague1.jpg"}
                
            elif style == "perm":
                if hair_length == "5":
                    hair = {"1": "애즈펌", "2": "댄디펌", "3": "리젠트펌",
                            "4": "쉐도우펌"}  # male, heart, perm, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Heart/perm/5/ads1.jpg",
                            "2": "./hair_image/recom_hair/man/Heart/perm/5/dandy1.jpg",
                            "3": "./hair_image/recom_hair/man/Heart/perm/5/regent1.jpg",
                            "4": "./hair_image/recom_hair/man/Heart/perm/5/shadow1.jpg"}

        elif face_shape == "긴형":
            if style == "cut":
                if hair_length == "5":
                    hair = {"1": "크롭컷", "2": "댄디컷", "3": "가일컷", "4": "아이비리그컷"}  # male, oblong, cut, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Oblong/cut/5/crop1.jpg",
                            "2": "./hair_image/recom_hair/man/Oblong/cut/5/dandy2.jpg",
                            "3": "./hair_image/recom_hair/man/Oblong/cut/5/gail2.jpg",
                            "4": "./hair_image/recom_hair/man/Oblong/cut/5/ivyleague2.jpg"}
            elif style == "perm":
                if hair_length == "5":
                    hair = {"1": "애즈펌", "2": "댄디펌", "3": "리젠트펌",
                            "4": "쉐도우펌"}  # male, oblong, perm, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Oblong/perm/5/ads2.jpg",
                            "2": "./hair_image/recom_hair/man/Oblong/perm/5/dandy2.jpg",
                            "3": "./hair_image/recom_hair/man/Oblong/perm/5/regent2.jpg",
                            "4": "./hair_image/recom_hair/man/Oblong/perm/5/shadow2.JPG"}

        elif face_shape == "둥근형":
            if style == "cut":
                if hair_length == "5":
                    hair = {"1": "크롭컷", "2": "댄디컷", "3": "가일컷", "4": "아이비리그컷"}  # male, round, cut, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Round/cut/5/crop3.JPG",
                            "2": "./hair_image/recom_hair/man/Round/cut/5/dandy5.JPG",
                            "3": "./hair_image/recom_hair/man/Round/cut/5/gail4.jpg",
                            "4": "./hair_image/recom_hair/man/Round/cut/5/ivyleague4.jfif"}
            elif style == "perm":
                if hair_length == "5":
                    hair = {"1": "애즈펌", "2": "댄디펌", "3": "리젠트펌",
                            "4": "쉐도우펌"}  # male, round, perm, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Round/perm/5/ads4.jpg",
                            "2": "./hair_image/recom_hair/man/Round/perm/5/dandy4.jpg",
                            "3": "./hair_image/recom_hair/man/Round/perm/5/regent4.jpg",
                            "4": "./hair_image/recom_hair/man/Round/perm/5/shadow4.jpg"}


        elif face_shape == "사각형":
            if style == "cut":
                if hair_length == "5":
                    hair = {"1": "댄디컷", "2": "플랫컷", "3": "아이비리그컷", "4": "리프컷"}  # male, square, cut, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Square/cut/5/dandy5.JPG",
                            "2": "./hair_image/recom_hair/man/Square/cut/5/flat1.jpg",
                            "3": "./hair_image/recom_hair/man/Square/cut/5/ivyleague4.jfif",
                            "4": "./hair_image/recom_hair/man/Square/cut/5/leaf1.jpg"}
            elif style == "perm":
                if hair_length == "5":
                    hair = {"1": "애즈펌", "2": "댄디펌", "3": "리젠트펌",
                            "4": "쉐도우펌"}  # male, square, perm, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Square/perm/5/ads5.jpg",
                            "2": "./hair_image/recom_hair/man/Square/perm/5/dandy5.JPG",
                            "3": "./hair_image/recom_hair/man/Square/perm/5/regent5.jfif",
                            "4": "./hair_image/recom_hair/man/Square/perm/5/shadow4.jpg"}


        elif face_shape == "타원형":
            if style == "cut":
                if hair_length == "5":
                    hair = {"1": "댄디컷", "2": "플랫컷", "3": "가일컷", "4": "리프컷"}  # male, oval, cut, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Oval/cut/5/dandy3.jpg",
                            "2": "./hair_image/recom_hair/man/Oval/cut/5/flat1.jpg",
                            "3": "./hair_image/recom_hair/man/Oval/cut/5/gail1.jpg",
                            "4": "./hair_image/recom_hair/man/Oval/cut/5/leaf2.JPG"}
            elif style == "perm":
                if hair_length == "5":
                    hair = {"1": "애즈펌", "2": "댄디펌", "3": "리젠트펌",
                            "4": "쉐도우펌"}  # male, oval, perm, 5 - 헤어스타일 4개 지정해서 넣어주기
                    path = {"1": "./hair_image/recom_hair/man/Oval/perm/5/ads3.pg",
                            "2": "./hair_image/recom_hair/man/Oval/perm/5/dandy3.jpg",
                            "3": "./hair_image/recom_hair/man/Oval/perm/5/regent3.jpg",
                            "4": "./hair_image/recom_hair/man/Oval/perm/5/shadow3.jpg"}

    return path, hair



### 도연님 이 밑으로 함수 넣어주세요 테스트 ###
def face_classify(name,phone_number,base_path,storage_path):
    from tensorflow.keras.models import load_model
    from tensorflow.python.keras.utils import np_utils
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Activation
    import tensorflow as tf
    import numpy as np
    import sys
    import os
    from numpy import argmax
    import matplotlib.pyplot as plt             
    import cv2                                 
    from tqdm import tqdm
    import shutil
    from pathlib import Path
    import json
    from collections import OrderedDict
    from tensorflow.keras import models
    from glob import glob
    import matplotlib.image as image

    import PIL
    import torch
    import efficientnet_pytorch
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    
    for (root, directories, files) in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            fp = file_path.split('/',4)[3].split('_',1)[0]

            if fp == name:

                # storage 폴더에 같은 '이름_전화번호_성별_컷/펌' 형식으로 폴더 생성 및 파일 복제
                storage = storage_path + name + '_' + phone_number
                os.mkdir(storage)
                shutil.copy(file_path, storage)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set gpu
                model = torch.load('face_model2.pt', map_location=torch.device('cpu')) # 모델 위치

                # 클래스 및 이미지 사이즈 지정하는 부분
                img = Image.open(file_path)
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                transform_norm = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Resize((224, 224)), transforms.Normalize(mean, std)])
#                 print(img)
                # get normalized image
                img_normalized = transform_norm(img).float()
#                 print(img_normalized)
                img_normalized = img_normalized.unsqueeze_(0)
                # input = Variable(image_tensor)
                img_normalized = img_normalized.to(device)
                # print(img_normalized.shape)
                
                with torch.no_grad():
                    model.eval()
                    output = model(img_normalized)
                    # print(output)
                    index = output.data.cpu().numpy().argmax()
                    classes = ['하트형', '긴형', '타원형', '둥근형', '사각형']
                    class_name = classes[index]
                    
                # 얼굴형 예측
                face_shape = class_name
                # ---------------------------------------------------------------------
                # 헤어스타일 예측
                hair_labels = ['리젠트','리프', '바디', '보니', '보브', '빌드', '소프트투블럭댄디', '숏단발', '쉐도우', 
                            '에어', '원랭스','플리츠', '허쉬', '히피','기타스타일']  #class 15
                
                length_labels = ['남자', '단발', '여숏', '장발', '중발']  #class 5

                # 모델 로드
                model_hair = models.load_model('./hair_class15_ver1.h5')
                model_length = models.load_model('./hair_length_class5_ver1.h5')
                
                img = image.imread(file_path)
                img_ = img[:, :, :3]
                img = cv2.resize(img_, (256, 256))
                img = img.reshape(1, 256, 256, 3)
                
                pred_hair = model_hair.predict(img)[0]
                pred_length = model_length.predict(img)[0]
                
                maxidx_hair = np.argmax(pred_hair)
                maxidx_length = np.argmax(pred_length)
                
                hair_style = f"{hair_labels[maxidx_hair]}"
                hair_length = f"{length_labels[maxidx_length]}"

                # 다 돌아갔으면 파일 지우기 --> 실행 X
                shutil.rmtree(base_path + name + '_' + phone_number)
    
                return face_shape, hair_style, hair_length
