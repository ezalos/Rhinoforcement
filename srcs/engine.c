/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   engine.c                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ezalos <ezalos@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/09/06 12:00:47 by ezalos            #+#    #+#             */
/*   Updated: 2019/09/30 00:01:57 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

int		is_game_finished(t_connect *c_four)
{
	int		r_v;

	if (c_four->turn < 7)
		r_v = FAILURE;
	else if (is_game_won(c_four) == SUCCESS)
		r_v = SUCCESS;
	else if (c_four->turn >= ROWS_NB * COLS_NB)
		r_v = SUCCESS;
	else
		r_v = FAILURE;
	if (r_v == SUCCESS)
	{
		c_four->end = SUCCESS;
		c_four->turn--;
		update_branch_data(c_four->actual_node, c_four->winner);
	}
	// DEBUG_INT(r_v);
	return (r_v);
}

int		play_move(t_connect *c_four, int move)
{
	static	int mv_tt = 0;
	int		row;
	int		col;
	int		r_v;

	r_v = FAILURE;
	col = move;
	if (col >= 0 && col < COLS_NB)
	{
		_C_CURSOR_SAVE;
		row = c_four->pile_size[col];
		if (row < ROWS_NB)
		{
			c_four->board[ROWS_NB - (row + 1)][col] = PLAYER_TURN(c_four);
			c_four->turn++;
			r_v = SUCCESS;
			// print_board(c_four);
			c_four->last_move = col;

			c_four->pile_size[col]++;
			if (c_four->actual_node->child[col] == NULL)
			{
				if (c_four->actual_node = create_node(c_four->actual_node, col, c_four->turn))
					r_v = SUCCESS;
			}
			else
				c_four->actual_node = c_four->actual_node->child[col];
			// ft_printf("%~{0;255;0}%d\n", ++mv_tt);
			_C_CURSOR_LOAD;
			if (c_four->player_type[0] == HUMAN
			||  c_four->player_type[1] == HUMAN)
				print_board(c_four);
		}
	}
	return (r_v);
}

//wins_son is doubled to allow storage of 1/2 for draws
//n_dad number of visits of parent node, n_son for child node being considered;
double	UCB1(int n_dad, int n_son, int wins_son)
{
	if (n_son == 0)
		return (1000000000000.0);
	return ((double)wins_son / ((double)n_son) + C_EXPLO * (sqrt(log(n_dad) / n_son))); //n_dad should be > 0 because n_son is > 0
}

void	play(t_connect *c_four)
{
	double		best;
	double		tmp;
	int		move;
	int		move_good;
	int		son;

	// printf("turn: %d\tPlayer: %d\n", c_four->turn, c_four->player_type[c_four->turn % 2]);
	if (c_four->player_type[c_four->turn % 2] == HUMAN)
		get_input(c_four);
	else
	{
		move_good = is_game_winnable(c_four);
		while (play_move(c_four, move_good) == FAILURE)
		{
			move = -1;
			best = -1;
			while (++move < COLS_NB)
			{
				tmp = -1000;
				if (c_four->pile_size[move] < ROWS_NB)
				{
					if (c_four->actual_node->child[move])
						son = TOTAL_DATA(c_four->actual_node->child[move]);
					else
						son = 0;
					if ((tmp = UCB1(TOTAL_DATA(c_four->actual_node), son, son ? c_four->actual_node->data[(c_four->turn % 2) + 1] : 0)) > best)
					{
						best = tmp;
						move_good = move;
					}
				}
				// printf("mv %d mvg%d tmp%d bst%d\n", move, move_good, tmp, best);
			}
		}

		//choose a moove from tree
		//go there and save data
		//
	}
}

int		engine(t_connect *c_four)
{
	// DEBUG_COLOR;
	init_new_game(c_four);
	if (c_four->player_type[0] == HUMAN
	||  c_four->player_type[1] == HUMAN)
		print_board(c_four);
	while (is_game_finished(c_four) == FAILURE)
	{
		play(c_four);
	}
	if (c_four->player_type[0] == HUMAN
	||  c_four->player_type[1] == HUMAN)
		print_board(c_four);
	// print_board(c_four);
	// ft_printf("\n. %d\tX %d\t%d O\n", c_four->tree->data[0], c_four->tree->data[1], c_four->tree->data[2]);
	return (c_four->winner);
}
