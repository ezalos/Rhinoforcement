/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   save_tree.c                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ezalos <ezalos@student.42.fr>              +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/09/28 15:37:32 by ezalos            #+#    #+#             */
/*   Updated: 2019/09/29 02:52:48 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

void		save_tree_to_file(t_monte_carlo *tree, int fd)
{
	char 	str[100];
	int		len;
	int		i;

	len = sprintf(str, ".%d X%d O%d\t", tree->data[0], tree->data[1], tree->data[2]);
	write(fd, str, len);
	i = -1;
	while (++i < COLS_NB)
	{
		if (tree->child[i])
			len = sprintf(str, "1");
		else
			len = sprintf(str, "0");
		write(fd, str, len);
	}
	write(fd, "\n", 1);
	i = -1;
	while (++i < COLS_NB)
		if (tree->child[i])
			save_tree_to_file(tree->child[i], fd);
}

void		load_tree_from_file(t_monte_carlo *tree, t_monte_carlo *daddy, int depth, int fd)
{
	// DEBUG_COLOR;
	char	*line;
	char	*line_2;
	char	*save;
	int		i;
	static int		prog = 0;

	tree->daddy = daddy;
	i = ft_gnl(fd, &line);
	// if (i < 1)
		// ft_printf("GNL %d\n", i);
	if ((i) > 0 )
	{
		// if (line[0] == '\n')
			// ft_printf("GNL \\n\n");
		// if (!line)
			// ft_printf("NULL\n");
		save = line;

		// printf("GNL Done\n");
		line_2 = ft_strchr(line, '.') + 1;
		tree->data[0] = (int)atoll(line_2);
		line_2 = ft_strchr(line, 'X') + 1;
		// ft_printf("%r%s", line, line_2);
		// ft_printf("\n", line, line_2);
		tree->data[1] = (int)atoll(line_2);
		line_2 = ft_strchr(line, 'O') + 1;
		tree->data[2] = (int)atoll(line_2);
		// printf("Atoi Done\n");
		line = ft_strchr(line, '\t') + 1;
		i = -1;
		while (++i < COLS_NB)
			if (line[i] == '1')
				tree->child[i] = ft_memalloc(sizeof(t_monte_carlo));
			else
				tree->child[i] = NULL;
		// printf("Malloc Done\n");
		ft_strdel(&save);
		// printf("Free Done\n");
		i = -1;
		while (++i < COLS_NB)
			if (tree->child[i])
			{
				if (prog == 0)
					ft_progress("LOAD", prog, COLS_NB * COLS_NB);
				load_tree_from_file(tree->child[i], tree, depth + 1, fd);
				if (depth == 2)
					ft_progress("LOAD", prog++, COLS_NB * COLS_NB);
				// printf("Recursive %d(%d) Done\n", depth, i);
			}
	}
}
