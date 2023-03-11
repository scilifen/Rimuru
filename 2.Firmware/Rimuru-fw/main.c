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

	gpio_init(KEY1, GPI, GPIO_LOW, GPI_PULL_UP); // 初始化 KEY1 输入 默认高电平 上拉输入
	gpio_init(KEY2, GPI, GPIO_LOW, GPI_PULL_UP); // 初始化 KEY1 输入 默认高电平 上拉输入
	gpio_init(KEY3, GPI, GPIO_LOW, GPI_PULL_UP); // 初始化 KEY1 输入 默认高电平 上拉输入
	gpio_init(KEY4, GPI, GPIO_LOW, GPI_PULL_UP); // 初始化 KEY1 输入 默认高电平 上拉输入

	gpio_init(SWITCH1, GPI, GPIO_HIGH, GPI_FLOATING_IN); // 初始化 SWITCH1 输入 默认高电平 浮空输入
	gpio_init(SWITCH2, GPI, GPIO_HIGH, GPI_FLOATING_IN);
	gpio_dir(SWITCH1, GPI, GPI_FLOATING_IN);
	gpio_dir(SWITCH2, GPI, GPI_FLOATING_IN);

	uart_init(UART_3, 115200, UART3_TX_B10, UART3_RX_B11);

	// pwm_init(TIM_2, TIM_2_CH1_A15, 100, 1.6 * 50000 / 10);
	// pwm_init(TIM_2, TIM_2_CH2_A01, 100, 0.7 * 50000);

	// pwm_enable(TIM_2);

	// pwm_disable(TIM_2);
	//
	// rt_thread_mdelay(2000);
	// while (1)
	// {

	// 	// pwm_duty_updata(TIM_2, TIM_2_CH1_A15, 1.90 * 50000 / 10);
	// 	// rt_thread_mdelay(2000);
	// 	// pwm_duty_updata(TIM_2, TIM_2_CH1_A15, 1.1 * 50000 / 10);
	// 	// rt_thread_mdelay(2000);
	// 	// rt_kprintf("%d\n", gpio_get(SWITCH1));
	// 	// rt_kprintf("%d\n", gpio_get(SWITCH2));
	// 	rt_kprintf("%d,%d,%d,%d,%d,%d,\n\r", gpio_get(KEY1), gpio_get(KEY2), gpio_get(KEY3), gpio_get(KEY4), gpio_get(SWITCH1), gpio_get(SWITCH2));
	// 	rt_thread_mdelay(100);
	// }
}