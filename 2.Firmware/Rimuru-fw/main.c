#include "headfile.h"

#define KEY1 (G0)
#define KEY2 (G1)
#define KEY3 (G2)
#define KEY4 (G3)

#define SWITCH1 (D14)
#define SWITCH2 (D15)

int main(void)
{
	gpio_init(H2, GPO, GPIO_LOW, GPO_PUSH_PULL);
	gpio_init(B13, GPO, GPIO_LOW, GPO_PUSH_PULL);

	uart_init(UART_3, 115200, UART3_TX_B10, UART3_RX_B11);

	pwm_init(TIM_2, TIM_2_CH1_A15, 100, 1.6 * 50000 / 10);
	pwm_init(TIM_2, TIM_2_CH2_A01, 100, 0.7 * 50000);

	pwm_enable(TIM_2);

	
	while (1)
	{
		pwm_duty_updata(TIM_2, TIM_2_CH1_A15, 1.90 * 50000 / 10);
		rt_thread_mdelay(2000);
		pwm_duty_updata(TIM_2, TIM_2_CH1_A15, 1.1 * 50000 / 10);
		rt_thread_mdelay(2000);
	}
}