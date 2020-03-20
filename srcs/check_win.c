/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   check_win.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ezalos <ezalos@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/09/27 21:53:18 by ezalos            #+#    #+#             */
/*   Updated: 2019/09/29 23:13:15 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

int		is_valid_position(int row, int row_mod, int col, int col_mod)
{
	int		r_v;

	r_v = FAILURE;
	if (row + row_mod >= 0 && row + row_mod < ROWS_NB)
		if (col + col_mod >= 0 && col + col_mod < COLS_NB)
			r_v = SUCCESS;
	return (r_v);
}

int		is_horizontal_win(t_connect *c_four, int player, int row, int col)
{
	int		r_v;
	int		col_mod;
	int		combo;

	combo = 0;
	r_v = FAILURE;
	col_mod = 1;
	while (	is_valid_position(row, 0,  col, col_mod) == SUCCESS
	&&		c_four->board[row][col + col_mod] == player)
	{
		combo++;
		col_mod++;
	}
	if (combo >= 3)
		r_v = SUCCESS;
	else
	{
		col_mod = -1;
		while (	is_valid_position(row, 0,  col, col_mod) == SUCCESS
		&&		c_four->board[row][col + col_mod] == player)
		{
			combo++;
			col_mod--;
		}
	}
	if (combo >= 3)
		r_v = SUCCESS;
	// DEBUG_INT(r_v);
	return (r_v);
}

int		is_vertical_win(t_connect *c_four, int player, int row, int col)
{
	int		r_v;
	int		row_mod;
	int		combo;

	combo = 0;
	r_v = FAILURE;
	row_mod = 1;
	while (	is_valid_position(row, row_mod,  col, 0) == SUCCESS
	&&		c_four->board[row + row_mod][col] == player)
	{
		combo++;
		row_mod++;
	}
	if (combo >= 3)
		r_v = SUCCESS;
	else
	{
		row_mod = -1;
		while (	is_valid_position(row, row_mod,  col, 0) == SUCCESS
		&&		c_four->board[row + row_mod][col] == player)
		{
			combo++;
			row_mod--;
		}
	}
	if (combo >= 3)
		r_v = SUCCESS;
	// DEBUG_INT(r_v);
	return (r_v);
}

int		is_diagonal_pos_win(t_connect *c_four, int player, int row, int col)
{
	int		r_v;
	int		row_mod;
	int		col_mod;
	int		combo;

	combo = 0;
	r_v = FAILURE;
	row_mod = 1;
	col_mod = 1;
	while (	is_valid_position(row, row_mod,  col, col_mod) == SUCCESS
	&&		c_four->board[row + row_mod][col + col_mod] == player)
	{
		combo++;
		row_mod++;
		col_mod++;
	}
	if (combo >= 3)
		r_v = SUCCESS;
	else
	{
		col_mod = -1;
		row_mod = -1;
		while (	is_valid_position(row, row_mod,  col, col_mod) == SUCCESS
		&&		c_four->board[row + row_mod][col + col_mod] == player)
		{
			combo++;
			row_mod--;
			col_mod--;
		}
	}
	if (combo >= 3)
		r_v = SUCCESS;
		// DEBUG_INT(r_v);
	return (r_v);
}

int		is_diagonal_neg_win(t_connect *c_four, int player, int row, int col)
{
	int		r_v;
	int		row_mod;
	int		col_mod;
	int		combo;

	combo = 0;
	r_v = FAILURE;
	row_mod = -1;
	col_mod = 1;
	while (	is_valid_position(row, row_mod,  col, col_mod) == SUCCESS
	&&		c_four->board[row + row_mod][col + col_mod] == player)
	{
		combo++;
		row_mod--;
		col_mod++;
	}
	if (combo >= 3)
		r_v = SUCCESS;
	else
	{
		col_mod = -1;
		row_mod = 1;
		while (	is_valid_position(row, row_mod,  col, col_mod) == SUCCESS
		&&		c_four->board[row + row_mod][col + col_mod] == player)
		{
			combo++;
			row_mod++;
			col_mod--;
		}
	}
	if (combo >= 3)
		r_v = SUCCESS;
		// DEBUG_INT(r_v);
	return (r_v);
}

int		is_game_won(t_connect *c_four)
{
	int		r_v;
	int		player;
	int		row;
	int		col;

	col = c_four->last_move;
	row = c_four->pile_size[col] - 1;
	row = ROWS_NB - (row + 1);
	player = c_four->board[row][col];
	// ft_printf("PLAYER %d\nX=%d\tO=%d\n", c_four->board[row][col], PLAYER_ONE, PLAYER_TWO);
	// ft_printf("Row %d\tCol %d\n", row, col);
	if ((r_v = is_horizontal_win(c_four, player, row, col)) == FAILURE)
		if ((r_v = is_vertical_win(c_four, player, row, col)) == FAILURE)
			if ((r_v = is_diagonal_pos_win(c_four, player, row, col)) == FAILURE)
				r_v = is_diagonal_neg_win(c_four, player, row, col);
	if (r_v == SUCCESS)
		c_four->winner = player;
	//DEBUG_INT(r_v);
	return (r_v);
}

int		is_game_winnable(t_connect *c_four)
{
	int		r_v;
	int		player;
	int		row;
	int		col;


	// ft_printf("PLAYER %d\nX=%d\tO=%d\n", c_four->board[row][col], PLAYER_ONE, PLAYER_TWO);
	// ft_printf("Row %d\tCol %d\n", row, col);
	r_v = FAILURE;
	col = -1;
	while (r_v == FAILURE && ++col < COLS_NB)
	{
		row = ROWS_NB - (c_four->pile_size[col] + 1);
		player = PLAYER_TURN(c_four);
		if ((r_v = is_horizontal_win(c_four, player, row, col)) == FAILURE)
			if ((r_v = is_vertical_win(c_four, player, row, col)) == FAILURE)
				if ((r_v = is_diagonal_pos_win(c_four, player, row, col)) == FAILURE)
					r_v = is_diagonal_neg_win(c_four, player, row, col);
	}
	//DEBUG_INT(r_v);
	return (col);
}
