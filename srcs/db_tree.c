/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   db_tree.c                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ezalos <ezalos@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/09/28 14:13:40 by ezalos            #+#    #+#             */
/*   Updated: 2019/09/28 16:10:11 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

t_monte_carlo		*create_node(t_monte_carlo *daddy, int col, int turn)
{
	static	int mv_tt = 0;
	int		r_v;

	// ft_printf("%d ", turn);
	if (col >= 0 && col < COLS_NB)
	{
		if (daddy->child[col] = ft_memalloc(sizeof(t_monte_carlo)))
		{
			daddy->child[col]->daddy = daddy;
			daddy->child[col]->depth = turn;
			ft_bzero(daddy->child[col]->child, sizeof(t_monte_carlo*) * COLS_NB);
			ft_bzero(daddy->child[col]->data, sizeof(int) * 3);
			// ft_printf("%~{0;0;255}%d ", ++mv_tt);
		}
	}
	return (daddy->child[col]);
}

void				update_branch_data(t_monte_carlo *daddy, int data)
{
	if (daddy)
	{
		if (data == PLAYER_NONE)
			daddy->data[0]++;
		else if (data == PLAYER_ONE)
			daddy->data[1]++;
		else if (data == PLAYER_TWO)
			daddy->data[2]++;
		update_branch_data(daddy->daddy, data);
	}
}
