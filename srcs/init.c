/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   init.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ezalos <ezalos@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/09/06 12:01:39 by ezalos            #+#    #+#             */
/*   Updated: 2019/09/29 21:55:17 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

void		init_new_game(t_connect *c_four)
{
	int		row;

	row = -1;
	while (++row < ROWS_NB)
		ft_memset(c_four->board[row], PLAYER_NONE, sizeof(char) * COLS_NB);
	row = -1;
	while (++row < COLS_NB)
		ft_bzero(c_four->pile_size, sizeof(int) * COLS_NB);
	c_four->turn = 0;
	c_four->winner = PLAYER_NONE;
	c_four->last_move = UNSET;
	c_four->last_last_move = UNSET;
	c_four->next_moove = UNSET;
	c_four->end = FAILURE;
	c_four->actual_node = c_four->tree;
}

void		init(t_connect *c_four)
{
	c_four->player_type[0] = COMPUTER;
	c_four->player_type[1] = COMPUTER;
	c_four->print = SUCCESS;
	c_four->last_save = 0;
	c_four->tree = ft_memalloc(sizeof(t_monte_carlo));
	ft_bzero(c_four->tree->child, sizeof(t_monte_carlo*) * COLS_NB);
	ft_bzero(c_four->tree->data, sizeof(int) * 3);
}
