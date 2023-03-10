/*
 * Copyright 2021 MindMotion Microelectronics Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "pin_init.h"
#include "hal_rcc.h"
#include "hal_gpio.h"

/*
 * UART1_TXD1 - PB6.
 * UART1_RXD1 - PB7.
 *
 * spi loopback:
 * SPI3_NSS  - PA15.
 * SPI3_MOSI - PC12.
 * SPI3_MISO - PC11.
 * SPI3_SCK  - PC10.
 *
 * spiflash:
 * SPI2_SS   - PB12.
 * SPI2_MISO - PB14.
 * SPI2_MOSI - PB15.
 * SPI2_SCK  - PB13.
 */
void BOARD_InitPins(void)
{
    /* PB6 - UART1_TX. */
    GPIO_Init_Type gpio_init;
    gpio_init.Pins  = GPIO_PIN_6;
    gpio_init.PinMode  = GPIO_PinMode_AF_PushPull;
    gpio_init.Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &gpio_init);
    GPIO_PinAFConf(GPIOB, gpio_init.Pins, GPIO_AF_7);

    /* PB7 - UART1_RX. */
    gpio_init.Pins  = GPIO_PIN_7;
    gpio_init.PinMode  = GPIO_PinMode_In_Floating;
    gpio_init.Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOB, &gpio_init);
    GPIO_PinAFConf(GPIOB, gpio_init.Pins, GPIO_AF_7);

    /* SPI3_NSS  - PA15. */
    gpio_init.Pins  = GPIO_PIN_15;
    gpio_init.PinMode  = GPIO_PinMode_AF_PushPull;
    gpio_init.Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &gpio_init);
    GPIO_PinAFConf(GPIOA, GPIO_PIN_15, GPIO_AF_6);

    /* SPI3_MOSI - PC12. */
    gpio_init.Pins  = GPIO_PIN_12;
    gpio_init.PinMode  = GPIO_PinMode_AF_PushPull;
    gpio_init.Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOC, &gpio_init);
    GPIO_PinAFConf(GPIOC, GPIO_PIN_12, GPIO_AF_6);

    /* SPI3_MISO - PC11. */
    gpio_init.Pins  = GPIO_PIN_11;
    gpio_init.PinMode  = GPIO_PinMode_In_Floating;
    gpio_init.Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOC, &gpio_init);
    GPIO_PinAFConf(GPIOC, GPIO_PIN_11, GPIO_AF_6);

    /* SPI3_SCK  - PC10. */
    gpio_init.Pins  = GPIO_PIN_10;
    gpio_init.PinMode  = GPIO_PinMode_AF_PushPull;
    gpio_init.Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOC, &gpio_init);
    GPIO_PinAFConf(GPIOC, GPIO_PIN_10, GPIO_AF_6);



    /* PB12 - SPI2_SS. */
    /* PB13 - SPI2_SCK. */
    /* PB14 - SPI2_MISO. */
    /* PB15 - SPI2_MOSI. */
}


/* EOF. */
