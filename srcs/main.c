/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.c                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: root <root@student.42.fr>                  +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2019/05/13 12:27:19 by root              #+#    #+#             */
/*   Updated: 2019/09/30 00:14:13 by ezalos           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "../includes/head.h"

int		main(int ac, char **av)
{
	t_connect	c_four;
	int			fd;
	int			exec = 0;

	init(&c_four);
	if (ac == 2)
		if ((fd = open(av[1], O_RDONLY)) > 0)
		{
			load_tree_from_file(c_four.tree, NULL, 0, fd);
			c_four.last_save = TOTAL_DATA(c_four.tree);
			ft_printf("Load complete\n");
			close(fd);
		}
	while (1)
	{
		if (exec > 100000)
		{
			_C_CURSOR_SAVE;
			printf("%d\n", TOTAL_DATA(c_four.tree));
			_C_CURSOR_LOAD;
			c_four.player_type[1] = HUMAN;
		}
		if (!(exec % 1000))
			printf("%d\n", exec);
		engine(&c_four);
		exec++;
		// c_four.player_type[0] = COMPUTER;
		if (TOTAL_DATA(c_four.tree) >= c_four.last_save + 1000000000)
		{
			print_board(&c_four);
			fd = ft_create_new_file(ft_strjoin("./data/", ft_nb_to_a(TOTAL_DATA(c_four.tree), 10)));
			if (fd > 0)
			{
				save_tree_to_file(c_four.tree, fd);
				c_four.last_save = TOTAL_DATA(c_four.tree);
				close(fd);
			}
			ft_printf("%~{250;150;250}Saved at: %d\n", TOTAL_DATA(c_four.tree));
			// c_four.player_type[0] = HUMAN;
		}
	}
}
