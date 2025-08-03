#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "logger.h"


char *get_time() {
    // Выделяем статический буфер для хранения строки времени
    static char time_str[30];

    // Получаем текущее время
    time_t raw_time;
    struct tm *time_info;
    time(&raw_time);
    time_info = localtime(&raw_time);

    // Получаем миллисекунды
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    long milliseconds = ts.tv_nsec / 1000000;

    // Форматируем строку времени
    strftime(time_str, sizeof(time_str), "%d-%m-%Y %H:%M:%S", time_info);

    return time_str;
}

void logger_info(char *s) {
    const char *time = get_time();

    printf("[%s] - INFO - %s", time, s);
}
