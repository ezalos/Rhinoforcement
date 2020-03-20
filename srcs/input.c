/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   input.c                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ezalos <ezalos@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/09/06 12:15:02 by ezalos            #+#    #+#             */
/*   Updated: 2019/09/29 18:30:07 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

void	fast_terminal(int on_off)
{
	static struct termios old = {0};

	if (on_off)
	{
		if (tcgetattr(0, &old) < 0)
			perror("tcsetattr()");
		old.c_lflag &= ~ICANON;
		old.c_lflag &= ~ECHO;
		old.c_cc[VMIN] = 1;
		old.c_cc[VTIME] = 0;
		if (tcsetattr(0, TCSANOW, &old) < 0)
			perror("tcsetattr ICANON");
	}
	else
	{
		old.c_lflag |= ICANON;
		old.c_lflag |= ECHO;
		if (tcsetattr(0, TCSADRAIN, &old) < 0)
			perror ("tcsetattr ~ICANON");
	}
}

int 	get_input(t_connect *c_four)
{
	// DEBUG_FUNC;
	char buffer[2];
	int	move;
	int	r_v;

	r_v = 1;
	// print_board(c_four);
	fast_terminal(1);
	move = UNSET;
	buffer[1] = '\0';
	while (move == UNSET && r_v)
	{
		// DEBUG_FUNC;
		if ((r_v = read(0, buffer, 1)) > 0)
		{
			// ft_printf("BUF->%c\n", buffer[0]);
			if (ft_isdigit(buffer[0]))
			{
				move = ft_atoi(buffer);
				// ft_printf("Atoi say: %d\n", move);
				if (play_move(c_four, move - 1) == FAILURE)
					move = UNSET;
			}
		}
		// ft_printf("OUT IF BUFF->%c\nmove: %d\tr_v: %d\n", buffer[0], move, r_v);
	}
	fast_terminal(0);
	return (r_v);
}
