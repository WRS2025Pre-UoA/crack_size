# crack_size
必要環境  
C++  
OpenCV  
LSDライブラリ（lsd.c, lsd.h）  

ファイル構成  
detect_size.cpp      # メインコード  
lsd.c            # LSDライブラリ本体  
lsd.h            # LSDライブラリヘッダ  
example.png      # （任意）入力画像の例  

ビルド方法  
g++ detect_size.cpp lsd.c  \`pkg-config --cflags --libs opencv4\`  

detect_size1.cpp 二値化あり　　
detect_size2.cpp　二値化なし←こっちを使う　　

実行方法  
./a.out 画像ファイル名.png
