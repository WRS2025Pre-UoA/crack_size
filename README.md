# crack_size
必要環境
C++
OpenCV
LSDライブラリ（lsd.c, lsd.h）

ファイル構成
├── detect4.cpp      # メインコード
├── lsd.c            # LSDライブラリ本体
├── lsd.h            # LSDライブラリヘッダ
└── example.png      # （任意）入力画像の例

ビルド方法
`g++ detect4.cpp lsd.c  pkg-config --cflags --libs opencv4`

実行方法
./a.out 画像ファイル名.png
