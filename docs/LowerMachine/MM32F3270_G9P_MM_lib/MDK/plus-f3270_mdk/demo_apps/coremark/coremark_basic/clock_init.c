/*
 * Copyright 2021 MindMotion Microelectronics Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "hal_common.h"
#include "clock_init.h"

#include "hal_rcc.h"

void CLOCK_ResetToDefault(void);
void CLOCK_BootToHSE120MHz(void);

void BOARD_InitBootClocks(void)
{
    CLOCK_ResetToDefault();
    CLOCK_BootToHSE120MHz();

    /* UART1. */
    RCC_EnableAPB2Periphs(RCC_APB2_PERIPH_UART1, true);
    RCC_ResetAPB2Periphs(RCC_APB2_PERIPH_UART1);

    /* GPIOB. */
    RCC_EnableAHB1Periphs(RCC_AHB1_PERIPH_GPIOB, true);
    RCC_ResetAHB1Periphs(RCC_AHB1_PERIPH_GPIOB);
}

/* Switch to HSI/6 = 8MHz. */
void CLOCK_ResetToDefault(void)
{
    /* Switch to HSI. */
    RCC->CR |= RCC_CR_HSION_MASK; /* Make sure the HSI is enabled. */
    while ( RCC_CR_HSIRDY_MASK != (RCC->CR & RCC_CR_HSIRDY_MASK) )
    {
    }
    RCC->CFGR = RCC_CFGR_SW(0u); /* Reset other clock sources and switch to HSI. */
    while ( RCC_CFGR_SWS(0u) != (RCC->CFGR & RCC_CFGR_SWS_MASK) ) /* Wait while the SYSCLK is switch to the HSI. */
    {
    }

    /* Reset all other clock sources. */
    RCC->CR = RCC_CR_HSION_MASK;

    /* Disable all interrupts and clear pending bits. */
    RCC->CIR = RCC->CIR; /* clear flags. */
    RCC->CIR = 0u; /* disable interrupts. */
}

/* Enable the PLL and use the HSE as input clock source. */
void CLOCK_BootToHSE120MHz(void)
{
    RCC->APB1ENR |= (1u << 28u); /* enable PWR/DBG. */
    PWR->CR1 = (PWR->CR1 & ~PWR_CR1_VOS_MASK) | PWR_CR1_VOS(1u); /* 1.65V. */

    /* enable HSE. */
    RCC->CR |= RCC_CR_HSEON_MASK;
    while ( RCC_CR_HSERDY_MASK != (RCC->CR & RCC_CR_HSERDY_MASK) )
    {
    }

    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC(1) /* (pllsrc == 1) ? HSE : HSI. */
                 | RCC_PLLCFGR_PLLMUL(19) /* (12 * (19 + 1)) / 2 = 120. */
                 | RCC_PLLCFGR_PLLDIV(1)
                 | RCC_PLLCFGR_PLLLDS(1)
                 | RCC_PLLCFGR_PLLICTRL(3)
                 ;

    /* Enable PLL. */
    RCC->CR |= RCC_CR_PLLON_MASK;
    while((RCC->CR & RCC_CR_PLLRDY_MASK) == 0)
    {
    }

    /* Enable the FLASH prefetch. */
    RCC->AHB1ENR |= (1u << 13u); /* enable the access to FLASH. */
    FLASH->ACR = FLASH_ACR_LATENCY(3u) /* setup divider. */
               | FLASH_ACR_PRFTBE_MASK /* enable flash prefetch. */
               ;

    /* Setup the dividers for each bus. */
    RCC->CFGR = RCC_CFGR_HPRE(0)   /* div=1 for AHB freq. */
              | RCC_CFGR_PPRE1(0x4)  /* div=2 for APB1 freq. */
              | RCC_CFGR_PPRE2(0x4)  /* div=2 for APB2 freq. */
              | RCC_CFGR_MCO(7)    /* use PLL/2 as output. */
              ;

    /* Switch the system clock source to PLL. */
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW_MASK) | RCC_CFGR_SW(2); /* use PLL as SYSCLK */


    /* Wait till PLL is used as system clock source. */
    while ( (RCC->CFGR & RCC_CFGR_SWS_MASK) != RCC_CFGR_SWS(2) )
    {
    }
}

/* EOF. */
