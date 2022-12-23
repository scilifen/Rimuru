/*
 * Copyright 2022 MindMotion Microelectronics Co., Ltd.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __CLOCK_INIT_H__
#define __CLOCK_INIT_H__

#define CLOCK_SYS_FREQ         8000000u
#define CLOCK_SYSTICK_FREQ     (CLOCK_SYS_FREQ/8u)
#define CLOCK_AHB1_FREQ        8000000u
#define CLOCK_AHB2_FREQ        8000000u
#define CLOCK_AHB3_FREQ        8000000u
#define CLOCK_APB1_FREQ        8000000u
#define CLOCK_APB2_FREQ        8000000u

void BOARD_InitBootClocks(void);
void CLOCK_ResetToDefault(void);
void CLOCK_SetClock_2M(void);

#endif /* __CLOCK_INIT_H__ */

