## Сборка

Сборка с оптимизациями:

```sh
сmake -DCMAKE_BUILD_TYPE=Release -B release
make --directory release
```

В директории `release/` будет несколько исполняемых файлов:


- тесты к библиотеке: `network_test`
- пример обучения на функции `sin`: `sin_train`
- обучения на `MNIST`: `mnist_train`