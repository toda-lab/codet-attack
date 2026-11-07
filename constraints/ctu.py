"""Constraints for the CTU dataset."""

import logging as _logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
    Constant,
    Feature,
    MathOperation,
)
from pyscipopt import Model, Variable
from typing_extensions import override
from z3 import And, ArithRef, Or, Solver, Sum

from smt_attack.box import BoundingBox
from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point

if TYPE_CHECKING:
    from collections.abc import Callable

logger = _logging.getLogger(__name__)

EPS = 1e-4


@dataclass(frozen=True)
class Item:
    idx: int
    min: float | None
    max: float | None
    mutable: bool
    type: Literal["real", "cat", "int", "date"]


@dataclass(frozen=True)
class CtuMeta:
    bytes_in_sum_s_1: Item = Item(0, 0.0, 2324.0, mutable=False, type="int")
    bytes_in_min_s_1: Item = Item(1, 0.0, 504.0, mutable=False, type="int")
    bytes_in_max_s_1: Item = Item(2, 0.0, 1820.0, mutable=False, type="int")
    bytes_out_sum_s_1: Item = Item(3, 0.0, 22690.0, mutable=True, type="int")
    bytes_out_min_s_1: Item = Item(4, 0.0, 22690.0, mutable=True, type="int")
    bytes_out_max_s_1: Item = Item(5, 0.0, 22690.0, mutable=True, type="int")
    pkts_in_sum_s_1: Item = Item(6, 0.0, 86.0, mutable=False, type="int")
    pkts_in_min_s_1: Item = Item(7, 0.0, 24.0, mutable=False, type="int")
    pkts_in_max_s_1: Item = Item(8, 0.0, 62.0, mutable=False, type="int")
    pkts_out_sum_s_1: Item = Item(9, 0.0, 122.0, mutable=True, type="int")
    pkts_out_min_s_1: Item = Item(10, 0.0, 122.0, mutable=True, type="int")
    pkts_out_max_s_1: Item = Item(11, 0.0, 122.0, mutable=True, type="int")
    duration_sum_s_1: Item = Item(12, 0.0, 923.0, mutable=True, type="real")
    duration_min_s_1: Item = Item(13, 0.0, 460.0, mutable=True, type="real")
    duration_max_s_1: Item = Item(14, 0.0, 463.0, mutable=True, type="real")
    tcp_sum_s_1: Item = Item(15, 0.0, 3571.0, mutable=True, type="int")
    udp_sum_s_1: Item = Item(16, 0.0, 1056.0, mutable=True, type="int")
    icmp_sum_s_1: Item = Item(17, 0.0, 22690.0, mutable=True, type="int")
    distinct_external_ips_s_1: Item = Item(18, 0.0, 8.0, mutable=False, type="int")
    distinct_src_port_s_1: Item = Item(19, 0.0, 2.0, mutable=False, type="int")
    distinct_dst_port_s_1: Item = Item(20, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_1: Item = Item(21, 0.0, 1893.0, mutable=False, type="int")
    bytes_in_min_d_1: Item = Item(22, 0.0, 1893.0, mutable=False, type="int")
    bytes_in_max_d_1: Item = Item(23, 0.0, 1893.0, mutable=False, type="int")
    bytes_out_sum_d_1: Item = Item(24, 0.0, 33964.0, mutable=True, type="int")
    bytes_out_min_d_1: Item = Item(25, 0.0, 33964.0, mutable=True, type="int")
    bytes_out_max_d_1: Item = Item(26, 0.0, 33964.0, mutable=True, type="int")
    pkts_in_sum_d_1: Item = Item(27, 0.0, 12.0, mutable=False, type="int")
    pkts_in_min_d_1: Item = Item(28, 0.0, 12.0, mutable=False, type="int")
    pkts_in_max_d_1: Item = Item(29, 0.0, 12.0, mutable=False, type="int")
    pkts_out_sum_d_1: Item = Item(30, 0.0, 1213.0, mutable=True, type="int")
    pkts_out_min_d_1: Item = Item(31, 0.0, 1213.0, mutable=True, type="int")
    pkts_out_max_d_1: Item = Item(32, 0.0, 1213.0, mutable=True, type="int")
    duration_sum_d_1: Item = Item(33, 0.0, 12330.0, mutable=True, type="real")
    duration_min_d_1: Item = Item(34, 0.0, 12330.0, mutable=True, type="real")
    duration_max_d_1: Item = Item(35, 0.0, 12330.0, mutable=True, type="real")
    tcp_sum_d_1: Item = Item(36, 0.0, 5042.0, mutable=True, type="int")
    udp_sum_d_1: Item = Item(37, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_1: Item = Item(38, 0.0, 33964.0, mutable=True, type="int")
    distinct_external_ips_d_1: Item = Item(39, 0.0, 20.0, mutable=False, type="int")
    distinct_src_port_d_1: Item = Item(40, 0.0, 2.0, mutable=False, type="int")
    distinct_dst_port_d_1: Item = Item(41, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_3: Item = Item(42, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_s_3: Item = Item(43, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_s_3: Item = Item(44, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_s_3: Item = Item(45, 0.0, 10272.0, mutable=True, type="int")
    bytes_out_min_s_3: Item = Item(46, 0.0, 6328.0, mutable=True, type="int")
    bytes_out_max_s_3: Item = Item(47, 0.0, 10272.0, mutable=True, type="int")
    pkts_in_sum_s_3: Item = Item(48, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_s_3: Item = Item(49, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_s_3: Item = Item(50, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_s_3: Item = Item(51, 0.0, 170.0, mutable=True, type="int")
    pkts_out_min_s_3: Item = Item(52, 0.0, 113.0, mutable=True, type="int")
    pkts_out_max_s_3: Item = Item(53, 0.0, 169.0, mutable=True, type="int")
    duration_sum_s_3: Item = Item(54, 0.0, 1316.0, mutable=True, type="real")
    duration_min_s_3: Item = Item(55, 0.0, 1169.0, mutable=True, type="real")
    duration_max_s_3: Item = Item(56, 0.0, 1169.0, mutable=True, type="real")
    tcp_sum_s_3: Item = Item(57, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_3: Item = Item(58, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_3: Item = Item(59, 0.0, 10272.0, mutable=True, type="int")
    distinct_external_ips_s_3: Item = Item(60, 0.0, 47.0, mutable=False, type="int")
    distinct_src_port_s_3: Item = Item(61, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_s_3: Item = Item(62, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_3: Item = Item(63, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_3: Item = Item(64, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_3: Item = Item(65, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_3: Item = Item(66, 0.0, 51185.0, mutable=True, type="int")
    bytes_out_min_d_3: Item = Item(67, 0.0, 18760.0, mutable=True, type="int")
    bytes_out_max_d_3: Item = Item(68, 0.0, 50312.0, mutable=True, type="int")
    pkts_in_sum_d_3: Item = Item(69, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_d_3: Item = Item(70, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_d_3: Item = Item(71, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_d_3: Item = Item(72, 0.0, 670.0, mutable=True, type="int")
    pkts_out_min_d_3: Item = Item(73, 0.0, 670.0, mutable=True, type="int")
    pkts_out_max_d_3: Item = Item(74, 0.0, 670.0, mutable=True, type="int")
    duration_sum_d_3: Item = Item(75, 0.0, 2702.0, mutable=True, type="real")
    duration_min_d_3: Item = Item(76, 0.0, 511.0, mutable=True, type="real")
    duration_max_d_3: Item = Item(77, 0.0, 2690.0, mutable=True, type="real")
    tcp_sum_d_3: Item = Item(78, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_3: Item = Item(79, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_3: Item = Item(80, 0.0, 51185.0, mutable=True, type="int")
    distinct_external_ips_d_3: Item = Item(81, 0.0, 34.0, mutable=False, type="int")
    distinct_src_port_d_3: Item = Item(82, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_d_3: Item = Item(83, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_8: Item = Item(84, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_s_8: Item = Item(85, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_s_8: Item = Item(86, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_s_8: Item = Item(87, 0.0, 320.0, mutable=True, type="int")
    bytes_out_min_s_8: Item = Item(88, 0.0, 320.0, mutable=True, type="int")
    bytes_out_max_s_8: Item = Item(89, 0.0, 320.0, mutable=True, type="int")
    pkts_in_sum_s_8: Item = Item(90, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_s_8: Item = Item(91, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_s_8: Item = Item(92, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_s_8: Item = Item(93, 0.0, 5.0, mutable=True, type="int")
    pkts_out_min_s_8: Item = Item(94, 0.0, 5.0, mutable=True, type="int")
    pkts_out_max_s_8: Item = Item(95, 0.0, 5.0, mutable=True, type="int")
    duration_sum_s_8: Item = Item(96, 0.0, 2.0, mutable=True, type="real")
    duration_min_s_8: Item = Item(97, 0.0, 2.0, mutable=True, type="real")
    duration_max_s_8: Item = Item(98, 0.0, 2.0, mutable=True, type="int")
    tcp_sum_s_8: Item = Item(99, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_8: Item = Item(100, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_8: Item = Item(101, 0.0, 320.0, mutable=True, type="int")
    distinct_external_ips_s_8: Item = Item(102, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_s_8: Item = Item(103, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_s_8: Item = Item(104, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_8: Item = Item(105, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_8: Item = Item(106, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_8: Item = Item(107, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_8: Item = Item(108, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_d_8: Item = Item(109, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_d_8: Item = Item(110, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_d_8: Item = Item(111, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_d_8: Item = Item(112, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_d_8: Item = Item(113, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_d_8: Item = Item(114, 0.0, 0.0, mutable=True, type="int")
    pkts_out_min_d_8: Item = Item(115, 0.0, 0.0, mutable=True, type="int")
    pkts_out_max_d_8: Item = Item(116, 0.0, 0.0, mutable=True, type="int")
    duration_sum_d_8: Item = Item(117, 0.0, 0.0, mutable=True, type="int")
    duration_min_d_8: Item = Item(118, 0.0, 0.0, mutable=True, type="int")
    duration_max_d_8: Item = Item(119, 0.0, 0.0, mutable=True, type="int")
    tcp_sum_d_8: Item = Item(120, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_8: Item = Item(121, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_8: Item = Item(122, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_8: Item = Item(123, 0.0, 0.0, mutable=False, type="int")
    distinct_src_port_d_8: Item = Item(124, 0.0, 0.0, mutable=False, type="int")
    distinct_dst_port_d_8: Item = Item(125, 0.0, 0.0, mutable=False, type="int")
    bytes_in_sum_s_10: Item = Item(126, 0.0, 287.0, mutable=False, type="int")
    bytes_in_min_s_10: Item = Item(127, 0.0, 287.0, mutable=False, type="int")
    bytes_in_max_s_10: Item = Item(128, 0.0, 287.0, mutable=False, type="int")
    bytes_out_sum_s_10: Item = Item(129, 0.0, 5376.0, mutable=True, type="int")
    bytes_out_min_s_10: Item = Item(130, 0.0, 5376.0, mutable=True, type="int")
    bytes_out_max_s_10: Item = Item(131, 0.0, 5376.0, mutable=True, type="int")
    pkts_in_sum_s_10: Item = Item(132, 0.0, 1.0, mutable=False, type="int")
    pkts_in_min_s_10: Item = Item(133, 0.0, 1.0, mutable=False, type="int")
    pkts_in_max_s_10: Item = Item(134, 0.0, 1.0, mutable=False, type="int")
    pkts_out_sum_s_10: Item = Item(135, 0.0, 112.0, mutable=True, type="int")
    pkts_out_min_s_10: Item = Item(136, 0.0, 112.0, mutable=True, type="int")
    pkts_out_max_s_10: Item = Item(137, 0.0, 112.0, mutable=True, type="int")
    duration_sum_s_10: Item = Item(138, 0.0, 1095.0, mutable=True, type="real")
    duration_min_s_10: Item = Item(139, 0.0, 1095.0, mutable=True, type="real")
    duration_max_s_10: Item = Item(140, 0.0, 1095.0, mutable=True, type="real")
    tcp_sum_s_10: Item = Item(141, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_10: Item = Item(142, 0.0, 390.0, mutable=True, type="int")
    icmp_sum_s_10: Item = Item(143, 0.0, 5376.0, mutable=True, type="int")
    distinct_external_ips_s_10: Item = Item(144, 0.0, 11.0, mutable=False, type="int")
    distinct_src_port_s_10: Item = Item(145, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_s_10: Item = Item(146, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_10: Item = Item(147, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_10: Item = Item(148, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_10: Item = Item(149, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_10: Item = Item(150, 0.0, 1769.0, mutable=True, type="int")
    bytes_out_min_d_10: Item = Item(151, 0.0, 1769.0, mutable=True, type="int")
    bytes_out_max_d_10: Item = Item(152, 0.0, 1769.0, mutable=True, type="int")
    pkts_in_sum_d_10: Item = Item(153, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_d_10: Item = Item(154, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_d_10: Item = Item(155, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_d_10: Item = Item(156, 0.0, 29.0, mutable=True, type="int")
    pkts_out_min_d_10: Item = Item(157, 0.0, 29.0, mutable=True, type="int")
    pkts_out_max_d_10: Item = Item(158, 0.0, 29.0, mutable=True, type="int")
    duration_sum_d_10: Item = Item(159, 0.0, 205.0, mutable=True, type="real")
    duration_min_d_10: Item = Item(160, 0.0, 205.0, mutable=True, type="real")
    duration_max_d_10: Item = Item(161, 0.0, 205.0, mutable=True, type="real")
    tcp_sum_d_10: Item = Item(162, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_10: Item = Item(163, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_10: Item = Item(164, 0.0, 1769.0, mutable=True, type="int")
    distinct_external_ips_d_10: Item = Item(165, 0.0, 3.0, mutable=False, type="int")
    distinct_src_port_d_10: Item = Item(166, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_d_10: Item = Item(167, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_21: Item = Item(168, 0.0, 13093.0, mutable=False, type="int")
    bytes_in_min_s_21: Item = Item(169, 0.0, 2918.0, mutable=False, type="int")
    bytes_in_max_s_21: Item = Item(170, 0.0, 2918.0, mutable=False, type="int")
    bytes_out_sum_s_21: Item = Item(171, 0.0, 1370.0, mutable=True, type="int")
    bytes_out_min_s_21: Item = Item(172, 0.0, 963.0, mutable=True, type="int")
    bytes_out_max_s_21: Item = Item(173, 0.0, 963.0, mutable=True, type="int")
    pkts_in_sum_s_21: Item = Item(174, 0.0, 647.0, mutable=False, type="int")
    pkts_in_min_s_21: Item = Item(175, 0.0, 60.0, mutable=False, type="int")
    pkts_in_max_s_21: Item = Item(176, 0.0, 124.0, mutable=False, type="int")
    pkts_out_sum_s_21: Item = Item(177, 0.0, 665.0, mutable=True, type="int")
    pkts_out_min_s_21: Item = Item(178, 0.0, 62.0, mutable=True, type="int")
    pkts_out_max_s_21: Item = Item(179, 0.0, 126.0, mutable=True, type="int")
    duration_sum_s_21: Item = Item(180, 0.0, 24811.0, mutable=True, type="real")
    duration_min_s_21: Item = Item(181, 0.0, 3403.0, mutable=True, type="real")
    duration_max_s_21: Item = Item(182, 0.0, 4970.0, mutable=True, type="real")
    tcp_sum_s_21: Item = Item(183, 0.0, 14463.0, mutable=True, type="int")
    udp_sum_s_21: Item = Item(184, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_21: Item = Item(185, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_21: Item = Item(186, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_s_21: Item = Item(187, 0.0, 8.0, mutable=False, type="int")
    distinct_dst_port_s_21: Item = Item(188, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_21: Item = Item(189, 0.0, 2074.0, mutable=False, type="int")
    bytes_in_min_d_21: Item = Item(190, 0.0, 2074.0, mutable=False, type="int")
    bytes_in_max_d_21: Item = Item(191, 0.0, 2074.0, mutable=False, type="int")
    bytes_out_sum_d_21: Item = Item(192, 0.0, 2415.0, mutable=True, type="int")
    bytes_out_min_d_21: Item = Item(193, 0.0, 2415.0, mutable=True, type="int")
    bytes_out_max_d_21: Item = Item(194, 0.0, 2415.0, mutable=True, type="int")
    pkts_in_sum_d_21: Item = Item(195, 0.0, 66.0, mutable=False, type="int")
    pkts_in_min_d_21: Item = Item(196, 0.0, 66.0, mutable=False, type="int")
    pkts_in_max_d_21: Item = Item(197, 0.0, 66.0, mutable=False, type="int")
    pkts_out_sum_d_21: Item = Item(198, 0.0, 69.0, mutable=True, type="int")
    pkts_out_min_d_21: Item = Item(199, 0.0, 65.0, mutable=True, type="int")
    pkts_out_max_d_21: Item = Item(200, 0.0, 65.0, mutable=True, type="int")
    duration_sum_d_21: Item = Item(201, 0.0, 949.0, mutable=True, type="real")
    duration_min_d_21: Item = Item(202, 0.0, 949.0, mutable=True, type="real")
    duration_max_d_21: Item = Item(203, 0.0, 949.0, mutable=True, type="real")
    tcp_sum_d_21: Item = Item(204, 0.0, 4489.0, mutable=True, type="int")
    udp_sum_d_21: Item = Item(205, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_21: Item = Item(206, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_21: Item = Item(207, 0.0, 2.0, mutable=False, type="int")
    distinct_src_port_d_21: Item = Item(208, 0.0, 13.0, mutable=False, type="int")
    distinct_dst_port_d_21: Item = Item(209, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_22: Item = Item(210, 0.0, 5504688.0, mutable=False, type="int")
    bytes_in_min_s_22: Item = Item(211, 0.0, 5504688.0, mutable=False, type="int")
    bytes_in_max_s_22: Item = Item(212, 0.0, 5504688.0, mutable=False, type="int")
    bytes_out_sum_s_22: Item = Item(213, 0.0, 36416071.0, mutable=True, type="int")
    bytes_out_min_s_22: Item = Item(214, 0.0, 36416071.0, mutable=True, type="int")
    bytes_out_max_s_22: Item = Item(215, 0.0, 36416071.0, mutable=True, type="int")
    pkts_in_sum_s_22: Item = Item(216, 0.0, 112871.0, mutable=False, type="int")
    pkts_in_min_s_22: Item = Item(217, 0.0, 112871.0, mutable=False, type="int")
    pkts_in_max_s_22: Item = Item(218, 0.0, 112871.0, mutable=False, type="int")
    pkts_out_sum_s_22: Item = Item(219, 0.0, 97183.0, mutable=True, type="int")
    pkts_out_min_s_22: Item = Item(220, 0.0, 97183.0, mutable=True, type="int")
    pkts_out_max_s_22: Item = Item(221, 0.0, 97183.0, mutable=True, type="int")
    duration_sum_s_22: Item = Item(222, 0.0, 21982.0, mutable=True, type="real")
    duration_min_s_22: Item = Item(223, 0.0, 21982.0, mutable=True, type="real")
    duration_max_s_22: Item = Item(224, 0.0, 21982.0, mutable=True, type="real")
    tcp_sum_s_22: Item = Item(225, 0.0, 36432973.0, mutable=True, type="int")
    udp_sum_s_22: Item = Item(226, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_22: Item = Item(227, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_22: Item = Item(228, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_s_22: Item = Item(229, 0.0, 4.0, mutable=False, type="int")
    distinct_dst_port_s_22: Item = Item(230, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_22: Item = Item(231, 0.0, 288089872.0, mutable=False, type="int")
    bytes_in_min_d_22: Item = Item(232, 0.0, 288089872.0, mutable=False, type="int")
    bytes_in_max_d_22: Item = Item(233, 0.0, 288089872.0, mutable=False, type="int")
    bytes_out_sum_d_22: Item = Item(234, 0.0, 9964136.0, mutable=True, type="int")
    bytes_out_min_d_22: Item = Item(235, 0.0, 2650380.0, mutable=True, type="int")
    bytes_out_max_d_22: Item = Item(236, 0.0, 9839688.0, mutable=True, type="int")
    pkts_in_sum_d_22: Item = Item(237, 0.0, 205248.0, mutable=False, type="int")
    pkts_in_min_d_22: Item = Item(238, 0.0, 205248.0, mutable=False, type="int")
    pkts_in_max_d_22: Item = Item(239, 0.0, 205248.0, mutable=False, type="int")
    pkts_out_sum_d_22: Item = Item(240, 0.0, 114041.0, mutable=True, type="int")
    pkts_out_min_d_22: Item = Item(241, 0.0, 114041.0, mutable=True, type="int")
    pkts_out_max_d_22: Item = Item(242, 0.0, 114041.0, mutable=True, type="int")
    duration_sum_d_22: Item = Item(243, 0.0, 33113.0, mutable=True, type="real")
    duration_min_d_22: Item = Item(244, 0.0, 21964.0, mutable=True, type="real")
    duration_max_d_22: Item = Item(245, 0.0, 22032.0, mutable=True, type="real")
    tcp_sum_d_22: Item = Item(246, 0.0, 288977872.0, mutable=True, type="int")
    udp_sum_d_22: Item = Item(247, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_22: Item = Item(248, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_22: Item = Item(249, 0.0, 2.0, mutable=False, type="int")
    distinct_src_port_d_22: Item = Item(250, 0.0, 54.0, mutable=False, type="int")
    distinct_dst_port_d_22: Item = Item(251, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_25: Item = Item(252, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_s_25: Item = Item(253, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_s_25: Item = Item(254, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_s_25: Item = Item(255, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_s_25: Item = Item(256, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_s_25: Item = Item(257, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_s_25: Item = Item(258, 0.0, 1.0, mutable=False, type="int")
    pkts_in_min_s_25: Item = Item(259, 0.0, 1.0, mutable=False, type="int")
    pkts_in_max_s_25: Item = Item(260, 0.0, 1.0, mutable=False, type="int")
    pkts_out_sum_s_25: Item = Item(261, 0.0, 85.0, mutable=True, type="int")
    pkts_out_min_s_25: Item = Item(262, 0.0, 2.0, mutable=True, type="int")
    pkts_out_max_s_25: Item = Item(263, 0.0, 2.0, mutable=True, type="int")
    duration_sum_s_25: Item = Item(264, 0.0, 94.0, mutable=True, type="real")
    duration_min_s_25: Item = Item(265, 0.0, 4.0, mutable=True, type="real")
    duration_max_s_25: Item = Item(266, 0.0, 5.0, mutable=True, type="real")
    tcp_sum_s_25: Item = Item(267, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_25: Item = Item(268, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_25: Item = Item(269, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_25: Item = Item(270, 0.0, 23.0, mutable=False, type="int")
    distinct_src_port_s_25: Item = Item(271, 0.0, 36.0, mutable=False, type="int")
    distinct_dst_port_s_25: Item = Item(272, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_25: Item = Item(273, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_25: Item = Item(274, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_25: Item = Item(275, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_25: Item = Item(276, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_d_25: Item = Item(277, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_d_25: Item = Item(278, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_d_25: Item = Item(279, 0.0, 12.0, mutable=False, type="int")
    pkts_in_min_d_25: Item = Item(280, 0.0, 12.0, mutable=False, type="int")
    pkts_in_max_d_25: Item = Item(281, 0.0, 12.0, mutable=False, type="int")
    pkts_out_sum_d_25: Item = Item(282, 0.0, 0.0, mutable=True, type="int")
    pkts_out_min_d_25: Item = Item(283, 0.0, 0.0, mutable=True, type="int")
    pkts_out_max_d_25: Item = Item(284, 0.0, 0.0, mutable=True, type="int")
    duration_sum_d_25: Item = Item(285, 0.0, 36.0, mutable=True, type="int")
    duration_min_d_25: Item = Item(286, 0.0, 36.0, mutable=True, type="int")
    duration_max_d_25: Item = Item(287, 0.0, 36.0, mutable=True, type="int")
    tcp_sum_d_25: Item = Item(288, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_25: Item = Item(289, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_25: Item = Item(290, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_25: Item = Item(291, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_d_25: Item = Item(292, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_d_25: Item = Item(293, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_53: Item = Item(294, 0.0, 169936.0, mutable=False, type="int")
    bytes_in_min_s_53: Item = Item(295, 0.0, 274.0, mutable=False, type="int")
    bytes_in_max_s_53: Item = Item(296, 0.0, 10542.0, mutable=False, type="int")
    bytes_out_sum_s_53: Item = Item(297, 0.0, 31727.0, mutable=True, type="int")
    bytes_out_min_s_53: Item = Item(298, 0.0, 220.0, mutable=True, type="int")
    bytes_out_max_s_53: Item = Item(299, 0.0, 6673.0, mutable=True, type="int")
    pkts_in_sum_s_53: Item = Item(300, 0.0, 946.0, mutable=False, type="int")
    pkts_in_min_s_53: Item = Item(301, 0.0, 3.0, mutable=False, type="int")
    pkts_in_max_s_53: Item = Item(302, 0.0, 46.0, mutable=False, type="int")
    pkts_out_sum_s_53: Item = Item(303, 0.0, 1031.0, mutable=True, type="int")
    pkts_out_min_s_53: Item = Item(304, 0.0, 5.0, mutable=True, type="int")
    pkts_out_max_s_53: Item = Item(305, 0.0, 221.0, mutable=True, type="int")
    duration_sum_s_53: Item = Item(306, 0.0, 909.0, mutable=True, type="real")
    duration_min_s_53: Item = Item(307, 0.0, 17.0, mutable=True, type="real")
    duration_max_s_53: Item = Item(308, 0.0, 473.0, mutable=True, type="real")
    tcp_sum_s_53: Item = Item(309, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_53: Item = Item(310, 0.0, 201663.0, mutable=True, type="int")
    icmp_sum_s_53: Item = Item(311, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_53: Item = Item(312, 0.0, 272.0, mutable=False, type="int")
    distinct_src_port_s_53: Item = Item(313, 0.0, 213.0, mutable=False, type="int")
    distinct_dst_port_s_53: Item = Item(314, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_53: Item = Item(315, 0.0, 4435.0, mutable=False, type="int")
    bytes_in_min_d_53: Item = Item(316, 0.0, 1805.0, mutable=False, type="int")
    bytes_in_max_d_53: Item = Item(317, 0.0, 1805.0, mutable=False, type="int")
    bytes_out_sum_d_53: Item = Item(318, 0.0, 255.0, mutable=True, type="int")
    bytes_out_min_d_53: Item = Item(319, 0.0, 255.0, mutable=True, type="int")
    bytes_out_max_d_53: Item = Item(320, 0.0, 255.0, mutable=True, type="int")
    pkts_in_sum_d_53: Item = Item(321, 0.0, 46.0, mutable=False, type="int")
    pkts_in_min_d_53: Item = Item(322, 0.0, 5.0, mutable=False, type="int")
    pkts_in_max_d_53: Item = Item(323, 0.0, 5.0, mutable=False, type="int")
    pkts_out_sum_d_53: Item = Item(324, 0.0, 5.0, mutable=True, type="int")
    pkts_out_min_d_53: Item = Item(325, 0.0, 5.0, mutable=True, type="int")
    pkts_out_max_d_53: Item = Item(326, 0.0, 5.0, mutable=True, type="int")
    duration_sum_d_53: Item = Item(327, 0.0, 141.0, mutable=True, type="real")
    duration_min_d_53: Item = Item(328, 0.0, 14.0, mutable=True, type="real")
    duration_max_d_53: Item = Item(329, 0.0, 15.0, mutable=True, type="real")
    tcp_sum_d_53: Item = Item(330, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_53: Item = Item(331, 0.0, 4435.0, mutable=True, type="int")
    icmp_sum_d_53: Item = Item(332, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_53: Item = Item(333, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_d_53: Item = Item(334, 0.0, 29.0, mutable=False, type="int")
    distinct_dst_port_d_53: Item = Item(335, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_80: Item = Item(336, 0.0, 4277426661.0, mutable=False, type="int")
    bytes_in_min_s_80: Item = Item(337, 0.0, 450243362.0, mutable=False, type="int")
    bytes_in_max_s_80: Item = Item(338, 0.0, 4276001650.0, mutable=False, type="int")
    bytes_out_sum_s_80: Item = Item(339, 0.0, 1999271276.0, mutable=True, type="int")
    bytes_out_min_s_80: Item = Item(340, 0.0, 1999271276.0, mutable=True, type="int")
    bytes_out_max_s_80: Item = Item(341, 0.0, 1999271276.0, mutable=True, type="int")
    pkts_in_sum_s_80: Item = Item(342, 0.0, 730585.0, mutable=False, type="int")
    pkts_in_min_s_80: Item = Item(343, 0.0, 522572.0, mutable=False, type="int")
    pkts_in_max_s_80: Item = Item(344, 0.0, 574569.0, mutable=False, type="int")
    pkts_out_sum_s_80: Item = Item(345, 0.0, 405463.0, mutable=True, type="int")
    pkts_out_min_s_80: Item = Item(346, 0.0, 291268.0, mutable=True, type="int")
    pkts_out_max_s_80: Item = Item(347, 0.0, 308941.0, mutable=True, type="int")
    duration_sum_s_80: Item = Item(348, 0.0, 105751.0, mutable=True, type="real")
    duration_min_s_80: Item = Item(349, 0.0, 21976.0, mutable=True, type="real")
    duration_max_s_80: Item = Item(350, 0.0, 21976.0, mutable=True, type="real")
    tcp_sum_s_80: Item = Item(351, 0.0, 4277608069.0, mutable=True, type="int")
    udp_sum_s_80: Item = Item(352, 0.0, 5129.0, mutable=True, type="int")
    icmp_sum_s_80: Item = Item(353, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_80: Item = Item(354, 0.0, 178.0, mutable=False, type="int")
    distinct_src_port_s_80: Item = Item(355, 0.0, 925.0, mutable=False, type="int")
    distinct_dst_port_s_80: Item = Item(356, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_80: Item = Item(357, 0.0, 10691181708.0, mutable=False, type="int")
    bytes_in_min_d_80: Item = Item(358, 0.0, 561398137.0, mutable=False, type="int")
    bytes_in_max_d_80: Item = Item(359, 0.0, 10691181708.0, mutable=False, type="int")
    bytes_out_sum_d_80: Item = Item(360, 0.0, 1949649056.0, mutable=True, type="int")
    bytes_out_min_d_80: Item = Item(361, 0.0, 364132.0, mutable=True, type="int")
    bytes_out_max_d_80: Item = Item(362, 0.0, 1949648324.0, mutable=True, type="int")
    pkts_in_sum_d_80: Item = Item(363, 0.0, 1312538.0, mutable=False, type="int")
    pkts_in_min_d_80: Item = Item(364, 0.0, 456537.0, mutable=False, type="int")
    pkts_in_max_d_80: Item = Item(365, 0.0, 456537.0, mutable=False, type="int")
    pkts_out_sum_d_80: Item = Item(366, 0.0, 380498.0, mutable=True, type="int")
    pkts_out_min_d_80: Item = Item(367, 0.0, 254178.0, mutable=True, type="int")
    pkts_out_max_d_80: Item = Item(368, 0.0, 254178.0, mutable=True, type="int")
    duration_sum_d_80: Item = Item(369, 0.0, 146718.0, mutable=True, type="real")
    duration_min_d_80: Item = Item(370, 0.0, 21801.0, mutable=True, type="real")
    duration_max_d_80: Item = Item(371, 0.0, 22016.0, mutable=True, type="real")
    tcp_sum_d_80: Item = Item(372, 0.0, 10691181743.0, mutable=True, type="int")
    udp_sum_d_80: Item = Item(373, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_80: Item = Item(374, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_80: Item = Item(375, 0.0, 29.0, mutable=False, type="int")
    distinct_src_port_d_80: Item = Item(376, 0.0, 94.0, mutable=False, type="int")
    distinct_dst_port_d_80: Item = Item(377, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_110: Item = Item(378, 0.0, 635850.0, mutable=False, type="int")
    bytes_in_min_s_110: Item = Item(379, 0.0, 401065.0, mutable=False, type="int")
    bytes_in_max_s_110: Item = Item(380, 0.0, 634521.0, mutable=False, type="int")
    bytes_out_sum_s_110: Item = Item(381, 0.0, 4828.0, mutable=True, type="int")
    bytes_out_min_s_110: Item = Item(382, 0.0, 907.0, mutable=True, type="int")
    bytes_out_max_s_110: Item = Item(383, 0.0, 3351.0, mutable=True, type="int")
    pkts_in_sum_s_110: Item = Item(384, 0.0, 859.0, mutable=False, type="int")
    pkts_in_min_s_110: Item = Item(385, 0.0, 619.0, mutable=False, type="int")
    pkts_in_max_s_110: Item = Item(386, 0.0, 630.0, mutable=False, type="int")
    pkts_out_sum_s_110: Item = Item(387, 0.0, 432.0, mutable=True, type="int")
    pkts_out_min_s_110: Item = Item(388, 0.0, 208.0, mutable=True, type="int")
    pkts_out_max_s_110: Item = Item(389, 0.0, 421.0, mutable=True, type="int")
    duration_sum_s_110: Item = Item(390, 0.0, 91.0, mutable=True, type="real")
    duration_min_s_110: Item = Item(391, 0.0, 91.0, mutable=True, type="real")
    duration_max_s_110: Item = Item(392, 0.0, 91.0, mutable=True, type="real")
    tcp_sum_s_110: Item = Item(393, 0.0, 637557.0, mutable=True, type="int")
    udp_sum_s_110: Item = Item(394, 0.0, 38.0, mutable=True, type="int")
    icmp_sum_s_110: Item = Item(395, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_110: Item = Item(396, 0.0, 4.0, mutable=False, type="int")
    distinct_src_port_s_110: Item = Item(397, 0.0, 6.0, mutable=False, type="int")
    distinct_dst_port_s_110: Item = Item(398, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_110: Item = Item(399, 0.0, 16510537.0, mutable=False, type="int")
    bytes_in_min_d_110: Item = Item(400, 0.0, 16510537.0, mutable=False, type="int")
    bytes_in_max_d_110: Item = Item(401, 0.0, 16510537.0, mutable=False, type="int")
    bytes_out_sum_d_110: Item = Item(402, 0.0, 4044.0, mutable=True, type="int")
    bytes_out_min_d_110: Item = Item(403, 0.0, 4044.0, mutable=True, type="int")
    bytes_out_max_d_110: Item = Item(404, 0.0, 4044.0, mutable=True, type="int")
    pkts_in_sum_d_110: Item = Item(405, 0.0, 12144.0, mutable=False, type="int")
    pkts_in_min_d_110: Item = Item(406, 0.0, 12144.0, mutable=False, type="int")
    pkts_in_max_d_110: Item = Item(407, 0.0, 12144.0, mutable=False, type="int")
    pkts_out_sum_d_110: Item = Item(408, 0.0, 6506.0, mutable=True, type="int")
    pkts_out_min_d_110: Item = Item(409, 0.0, 6506.0, mutable=True, type="int")
    pkts_out_max_d_110: Item = Item(410, 0.0, 6506.0, mutable=True, type="int")
    duration_sum_d_110: Item = Item(411, 0.0, 1159.0, mutable=True, type="real")
    duration_min_d_110: Item = Item(412, 0.0, 1091.0, mutable=True, type="real")
    duration_max_d_110: Item = Item(413, 0.0, 1159.0, mutable=True, type="real")
    tcp_sum_d_110: Item = Item(414, 0.0, 16514581.0, mutable=True, type="int")
    udp_sum_d_110: Item = Item(415, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_110: Item = Item(416, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_110: Item = Item(417, 0.0, 2.0, mutable=False, type="int")
    distinct_src_port_d_110: Item = Item(418, 0.0, 2.0, mutable=False, type="int")
    distinct_dst_port_d_110: Item = Item(419, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_123: Item = Item(420, 0.0, 66000.0, mutable=False, type="int")
    bytes_in_min_s_123: Item = Item(421, 0.0, 33024.0, mutable=False, type="int")
    bytes_in_max_s_123: Item = Item(422, 0.0, 33024.0, mutable=False, type="int")
    bytes_out_sum_s_123: Item = Item(423, 0.0, 66000.0, mutable=True, type="int")
    bytes_out_min_s_123: Item = Item(424, 0.0, 33024.0, mutable=True, type="int")
    bytes_out_max_s_123: Item = Item(425, 0.0, 33024.0, mutable=True, type="int")
    pkts_in_sum_s_123: Item = Item(426, 0.0, 1375.0, mutable=False, type="int")
    pkts_in_min_s_123: Item = Item(427, 0.0, 688.0, mutable=False, type="int")
    pkts_in_max_s_123: Item = Item(428, 0.0, 688.0, mutable=False, type="int")
    pkts_out_sum_s_123: Item = Item(429, 0.0, 1375.0, mutable=True, type="int")
    pkts_out_min_s_123: Item = Item(430, 0.0, 688.0, mutable=True, type="int")
    pkts_out_max_s_123: Item = Item(431, 0.0, 688.0, mutable=True, type="int")
    duration_sum_s_123: Item = Item(432, 0.0, 43993.0, mutable=True, type="real")
    duration_min_s_123: Item = Item(433, 0.0, 22019.0, mutable=True, type="real")
    duration_max_s_123: Item = Item(434, 0.0, 22019.0, mutable=True, type="real")
    tcp_sum_s_123: Item = Item(435, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_123: Item = Item(436, 0.0, 132000.0, mutable=True, type="int")
    icmp_sum_s_123: Item = Item(437, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_123: Item = Item(438, 0.0, 17.0, mutable=False, type="int")
    distinct_src_port_s_123: Item = Item(439, 0.0, 7.0, mutable=False, type="int")
    distinct_dst_port_s_123: Item = Item(440, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_123: Item = Item(441, 0.0, 48.0, mutable=False, type="int")
    bytes_in_min_d_123: Item = Item(442, 0.0, 48.0, mutable=False, type="int")
    bytes_in_max_d_123: Item = Item(443, 0.0, 48.0, mutable=False, type="int")
    bytes_out_sum_d_123: Item = Item(444, 0.0, 136.0, mutable=True, type="int")
    bytes_out_min_d_123: Item = Item(445, 0.0, 136.0, mutable=True, type="int")
    bytes_out_max_d_123: Item = Item(446, 0.0, 136.0, mutable=True, type="int")
    pkts_in_sum_d_123: Item = Item(447, 0.0, 1.0, mutable=False, type="int")
    pkts_in_min_d_123: Item = Item(448, 0.0, 1.0, mutable=False, type="int")
    pkts_in_max_d_123: Item = Item(449, 0.0, 1.0, mutable=False, type="int")
    pkts_out_sum_d_123: Item = Item(450, 0.0, 2.0, mutable=True, type="int")
    pkts_out_min_d_123: Item = Item(451, 0.0, 2.0, mutable=True, type="int")
    pkts_out_max_d_123: Item = Item(452, 0.0, 2.0, mutable=True, type="int")
    duration_sum_d_123: Item = Item(453, 0.0, 33.0, mutable=True, type="real")
    duration_min_d_123: Item = Item(454, 0.0, 33.0, mutable=True, type="real")
    duration_max_d_123: Item = Item(455, 0.0, 33.0, mutable=True, type="real")
    tcp_sum_d_123: Item = Item(456, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_123: Item = Item(457, 0.0, 136.0, mutable=True, type="int")
    icmp_sum_d_123: Item = Item(458, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_123: Item = Item(459, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_d_123: Item = Item(460, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_d_123: Item = Item(461, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_135: Item = Item(462, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_s_135: Item = Item(463, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_s_135: Item = Item(464, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_s_135: Item = Item(465, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_s_135: Item = Item(466, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_s_135: Item = Item(467, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_s_135: Item = Item(468, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_s_135: Item = Item(469, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_s_135: Item = Item(470, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_s_135: Item = Item(471, 0.0, 3.0, mutable=True, type="int")
    pkts_out_min_s_135: Item = Item(472, 0.0, 2.0, mutable=True, type="int")
    pkts_out_max_s_135: Item = Item(473, 0.0, 2.0, mutable=True, type="int")
    duration_sum_s_135: Item = Item(474, 0.0, 4.0, mutable=True, type="int")
    duration_min_s_135: Item = Item(475, 0.0, 4.0, mutable=True, type="int")
    duration_max_s_135: Item = Item(476, 0.0, 4.0, mutable=True, type="int")
    tcp_sum_s_135: Item = Item(477, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_135: Item = Item(478, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_135: Item = Item(479, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_135: Item = Item(480, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_s_135: Item = Item(481, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_s_135: Item = Item(482, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_135: Item = Item(483, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_135: Item = Item(484, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_135: Item = Item(485, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_135: Item = Item(486, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_d_135: Item = Item(487, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_d_135: Item = Item(488, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_d_135: Item = Item(489, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_d_135: Item = Item(490, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_d_135: Item = Item(491, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_d_135: Item = Item(492, 0.0, 0.0, mutable=True, type="int")
    pkts_out_min_d_135: Item = Item(493, 0.0, 0.0, mutable=True, type="int")
    pkts_out_max_d_135: Item = Item(494, 0.0, 0.0, mutable=True, type="int")
    duration_sum_d_135: Item = Item(495, 0.0, 0.0, mutable=True, type="int")
    duration_min_d_135: Item = Item(496, 0.0, 0.0, mutable=True, type="int")
    duration_max_d_135: Item = Item(497, 0.0, 0.0, mutable=True, type="int")
    tcp_sum_d_135: Item = Item(498, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_135: Item = Item(499, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_135: Item = Item(500, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_135: Item = Item(501, 0.0, 0.0, mutable=False, type="int")
    distinct_src_port_d_135: Item = Item(502, 0.0, 0.0, mutable=False, type="int")
    distinct_dst_port_d_135: Item = Item(503, 0.0, 0.0, mutable=False, type="int")
    bytes_in_sum_s_138: Item = Item(504, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_s_138: Item = Item(505, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_s_138: Item = Item(506, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_s_138: Item = Item(507, 0.0, 412.0, mutable=True, type="int")
    bytes_out_min_s_138: Item = Item(508, 0.0, 412.0, mutable=True, type="int")
    bytes_out_max_s_138: Item = Item(509, 0.0, 412.0, mutable=True, type="int")
    pkts_in_sum_s_138: Item = Item(510, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_s_138: Item = Item(511, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_s_138: Item = Item(512, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_s_138: Item = Item(513, 0.0, 2.0, mutable=True, type="int")
    pkts_out_min_s_138: Item = Item(514, 0.0, 2.0, mutable=True, type="int")
    pkts_out_max_s_138: Item = Item(515, 0.0, 2.0, mutable=True, type="int")
    duration_sum_s_138: Item = Item(516, 0.0, 1.0, mutable=True, type="real")
    duration_min_s_138: Item = Item(517, 0.0, 1.0, mutable=True, type="real")
    duration_max_s_138: Item = Item(518, 0.0, 1.0, mutable=True, type="real")
    tcp_sum_s_138: Item = Item(519, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_138: Item = Item(520, 0.0, 412.0, mutable=True, type="int")
    icmp_sum_s_138: Item = Item(521, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_138: Item = Item(522, 0.0, 1.0, mutable=False, type="int")
    distinct_src_port_s_138: Item = Item(523, 0.0, 1.0, mutable=False, type="int")
    distinct_dst_port_s_138: Item = Item(524, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_138: Item = Item(525, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_138: Item = Item(526, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_138: Item = Item(527, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_138: Item = Item(528, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_d_138: Item = Item(529, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_d_138: Item = Item(530, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_d_138: Item = Item(531, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_d_138: Item = Item(532, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_d_138: Item = Item(533, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_d_138: Item = Item(534, 0.0, 0.0, mutable=True, type="int")
    pkts_out_min_d_138: Item = Item(535, 0.0, 0.0, mutable=True, type="int")
    pkts_out_max_d_138: Item = Item(536, 0.0, 0.0, mutable=True, type="int")
    duration_sum_d_138: Item = Item(537, 0.0, 0.0, mutable=True, type="int")
    duration_min_d_138: Item = Item(538, 0.0, 0.0, mutable=True, type="int")
    duration_max_d_138: Item = Item(539, 0.0, 0.0, mutable=True, type="int")
    tcp_sum_d_138: Item = Item(540, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_138: Item = Item(541, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_138: Item = Item(542, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_138: Item = Item(543, 0.0, 0.0, mutable=False, type="int")
    distinct_src_port_d_138: Item = Item(544, 0.0, 0.0, mutable=False, type="int")
    distinct_dst_port_d_138: Item = Item(545, 0.0, 0.0, mutable=False, type="int")
    bytes_in_sum_s_161: Item = Item(546, 0.0, 196.0, mutable=False, type="int")
    bytes_in_min_s_161: Item = Item(547, 0.0, 83.0, mutable=False, type="int")
    bytes_in_max_s_161: Item = Item(548, 0.0, 113.0, mutable=False, type="int")
    bytes_out_sum_s_161: Item = Item(549, 0.0, 257400.0, mutable=True, type="int")
    bytes_out_min_s_161: Item = Item(550, 0.0, 257400.0, mutable=True, type="int")
    bytes_out_max_s_161: Item = Item(551, 0.0, 257400.0, mutable=True, type="int")
    pkts_in_sum_s_161: Item = Item(552, 0.0, 3.0, mutable=False, type="int")
    pkts_in_min_s_161: Item = Item(553, 0.0, 1.0, mutable=False, type="int")
    pkts_in_max_s_161: Item = Item(554, 0.0, 2.0, mutable=False, type="int")
    pkts_out_sum_s_161: Item = Item(555, 0.0, 3300.0, mutable=True, type="int")
    pkts_out_min_s_161: Item = Item(556, 0.0, 3300.0, mutable=True, type="int")
    pkts_out_max_s_161: Item = Item(557, 0.0, 3300.0, mutable=True, type="int")
    duration_sum_s_161: Item = Item(558, 0.0, 39962.0, mutable=True, type="real")
    duration_min_s_161: Item = Item(559, 0.0, 22021.0, mutable=True, type="real")
    duration_max_s_161: Item = Item(560, 0.0, 22021.0, mutable=True, type="real")
    tcp_sum_s_161: Item = Item(561, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_161: Item = Item(562, 0.0, 257400.0, mutable=True, type="int")
    icmp_sum_s_161: Item = Item(563, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_161: Item = Item(564, 0.0, 3.0, mutable=False, type="int")
    distinct_src_port_s_161: Item = Item(565, 0.0, 3.0, mutable=False, type="int")
    distinct_dst_port_s_161: Item = Item(566, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_161: Item = Item(567, 0.0, 39123.0, mutable=False, type="int")
    bytes_in_min_d_161: Item = Item(568, 0.0, 39123.0, mutable=False, type="int")
    bytes_in_max_d_161: Item = Item(569, 0.0, 39123.0, mutable=False, type="int")
    bytes_out_sum_d_161: Item = Item(570, 0.0, 228065.0, mutable=True, type="int")
    bytes_out_min_d_161: Item = Item(571, 0.0, 84006.0, mutable=True, type="int")
    bytes_out_max_d_161: Item = Item(572, 0.0, 98094.0, mutable=True, type="int")
    pkts_in_sum_d_161: Item = Item(573, 0.0, 483.0, mutable=False, type="int")
    pkts_in_min_d_161: Item = Item(574, 0.0, 483.0, mutable=False, type="int")
    pkts_in_max_d_161: Item = Item(575, 0.0, 483.0, mutable=False, type="int")
    pkts_out_sum_d_161: Item = Item(576, 0.0, 2846.0, mutable=True, type="int")
    pkts_out_min_d_161: Item = Item(577, 0.0, 1077.0, mutable=True, type="int")
    pkts_out_max_d_161: Item = Item(578, 0.0, 1212.0, mutable=True, type="int")
    duration_sum_d_161: Item = Item(579, 0.0, 47716.0, mutable=True, type="real")
    duration_min_d_161: Item = Item(580, 0.0, 21543.0, mutable=True, type="real")
    duration_max_d_161: Item = Item(581, 0.0, 21983.0, mutable=True, type="real")
    tcp_sum_d_161: Item = Item(582, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_161: Item = Item(583, 0.0, 228065.0, mutable=True, type="int")
    icmp_sum_d_161: Item = Item(584, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_161: Item = Item(585, 0.0, 4.0, mutable=False, type="int")
    distinct_src_port_d_161: Item = Item(586, 0.0, 4.0, mutable=False, type="int")
    distinct_dst_port_d_161: Item = Item(587, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_443: Item = Item(588, 0.0, 1280634620.0, mutable=False, type="int")
    bytes_in_min_s_443: Item = Item(589, 0.0, 2500262.0, mutable=False, type="int")
    bytes_in_max_s_443: Item = Item(590, 0.0, 1280390612.0, mutable=False, type="int")
    bytes_out_sum_s_443: Item = Item(591, 0.0, 92617457.0, mutable=True, type="int")
    bytes_out_min_s_443: Item = Item(592, 0.0, 17019941.0, mutable=True, type="int")
    bytes_out_max_s_443: Item = Item(593, 0.0, 45501654.0, mutable=True, type="int")
    pkts_in_sum_s_443: Item = Item(594, 0.0, 912375.0, mutable=False, type="int")
    pkts_in_min_s_443: Item = Item(595, 0.0, 9663.0, mutable=False, type="int")
    pkts_in_max_s_443: Item = Item(596, 0.0, 911788.0, mutable=False, type="int")
    pkts_out_sum_s_443: Item = Item(597, 0.0, 438072.0, mutable=True, type="int")
    pkts_out_min_s_443: Item = Item(598, 0.0, 17464.0, mutable=True, type="int")
    pkts_out_max_s_443: Item = Item(599, 0.0, 437677.0, mutable=True, type="int")
    duration_sum_s_443: Item = Item(600, 0.0, 417008.0, mutable=True, type="real")
    duration_min_s_443: Item = Item(601, 0.0, 22022.0, mutable=True, type="real")
    duration_max_s_443: Item = Item(602, 0.0, 22027.0, mutable=True, type="real")
    tcp_sum_s_443: Item = Item(603, 0.0, 1280829342.0, mutable=True, type="int")
    udp_sum_s_443: Item = Item(604, 0.0, 153434486.0, mutable=True, type="int")
    icmp_sum_s_443: Item = Item(605, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_443: Item = Item(606, 0.0, 55.0, mutable=False, type="int")
    distinct_src_port_s_443: Item = Item(607, 0.0, 283.0, mutable=False, type="int")
    distinct_dst_port_s_443: Item = Item(608, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_443: Item = Item(609, 0.0, 9662724483.0, mutable=False, type="int")
    bytes_in_min_d_443: Item = Item(610, 0.0, 82021574.0, mutable=False, type="int")
    bytes_in_max_d_443: Item = Item(611, 0.0, 9662697547.0, mutable=False, type="int")
    bytes_out_sum_d_443: Item = Item(612, 0.0, 14411004278.0, mutable=True, type="int")
    bytes_out_min_d_443: Item = Item(613, 0.0, 24483588.0, mutable=True, type="int")
    bytes_out_max_d_443: Item = Item(614, 0.0, 14410956052.0, mutable=True, type="int")
    pkts_in_sum_d_443: Item = Item(615, 0.0, 421005.0, mutable=False, type="int")
    pkts_in_min_d_443: Item = Item(616, 0.0, 421005.0, mutable=False, type="int")
    pkts_in_max_d_443: Item = Item(617, 0.0, 421005.0, mutable=False, type="int")
    pkts_out_sum_d_443: Item = Item(618, 0.0, 232541.0, mutable=True, type="int")
    pkts_out_min_d_443: Item = Item(619, 0.0, 208542.0, mutable=True, type="int")
    pkts_out_max_d_443: Item = Item(620, 0.0, 208542.0, mutable=True, type="int")
    duration_sum_d_443: Item = Item(621, 0.0, 862285.0, mutable=True, type="real")
    duration_min_d_443: Item = Item(622, 0.0, 22019.0, mutable=True, type="real")
    duration_max_d_443: Item = Item(623, 0.0, 22035.0, mutable=True, type="real")
    tcp_sum_d_443: Item = Item(624, 0.0, 14411015070.0, mutable=True, type="int")
    udp_sum_d_443: Item = Item(625, 0.0, 2264.0, mutable=True, type="int")
    icmp_sum_d_443: Item = Item(626, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_443: Item = Item(627, 0.0, 154.0, mutable=False, type="int")
    distinct_src_port_d_443: Item = Item(628, 0.0, 155.0, mutable=False, type="int")
    distinct_dst_port_d_443: Item = Item(629, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_445: Item = Item(630, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_s_445: Item = Item(631, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_s_445: Item = Item(632, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_s_445: Item = Item(633, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_s_445: Item = Item(634, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_s_445: Item = Item(635, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_s_445: Item = Item(636, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_s_445: Item = Item(637, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_s_445: Item = Item(638, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_s_445: Item = Item(639, 0.0, 14.0, mutable=True, type="int")
    pkts_out_min_s_445: Item = Item(640, 0.0, 2.0, mutable=True, type="int")
    pkts_out_max_s_445: Item = Item(641, 0.0, 2.0, mutable=True, type="int")
    duration_sum_s_445: Item = Item(642, 0.0, 16.0, mutable=True, type="real")
    duration_min_s_445: Item = Item(643, 0.0, 4.0, mutable=True, type="real")
    duration_max_s_445: Item = Item(644, 0.0, 4.0, mutable=True, type="real")
    tcp_sum_s_445: Item = Item(645, 0.0, 0.0, mutable=True, type="int")
    udp_sum_s_445: Item = Item(646, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_445: Item = Item(647, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_445: Item = Item(648, 0.0, 5.0, mutable=False, type="int")
    distinct_src_port_s_445: Item = Item(649, 0.0, 5.0, mutable=False, type="int")
    distinct_dst_port_s_445: Item = Item(650, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_445: Item = Item(651, 0.0, 0.0, mutable=False, type="int")
    bytes_in_min_d_445: Item = Item(652, 0.0, 0.0, mutable=False, type="int")
    bytes_in_max_d_445: Item = Item(653, 0.0, 0.0, mutable=False, type="int")
    bytes_out_sum_d_445: Item = Item(654, 0.0, 0.0, mutable=True, type="int")
    bytes_out_min_d_445: Item = Item(655, 0.0, 0.0, mutable=True, type="int")
    bytes_out_max_d_445: Item = Item(656, 0.0, 0.0, mutable=True, type="int")
    pkts_in_sum_d_445: Item = Item(657, 0.0, 0.0, mutable=False, type="int")
    pkts_in_min_d_445: Item = Item(658, 0.0, 0.0, mutable=False, type="int")
    pkts_in_max_d_445: Item = Item(659, 0.0, 0.0, mutable=False, type="int")
    pkts_out_sum_d_445: Item = Item(660, 0.0, 0.0, mutable=True, type="int")
    pkts_out_min_d_445: Item = Item(661, 0.0, 0.0, mutable=True, type="int")
    pkts_out_max_d_445: Item = Item(662, 0.0, 0.0, mutable=True, type="int")
    duration_sum_d_445: Item = Item(663, 0.0, 0.0, mutable=True, type="int")
    duration_min_d_445: Item = Item(664, 0.0, 0.0, mutable=True, type="int")
    duration_max_d_445: Item = Item(665, 0.0, 0.0, mutable=True, type="int")
    tcp_sum_d_445: Item = Item(666, 0.0, 0.0, mutable=True, type="int")
    udp_sum_d_445: Item = Item(667, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_445: Item = Item(668, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_445: Item = Item(669, 0.0, 0.0, mutable=False, type="int")
    distinct_src_port_d_445: Item = Item(670, 0.0, 0.0, mutable=False, type="int")
    distinct_dst_port_d_445: Item = Item(671, 0.0, 0.0, mutable=False, type="int")
    bytes_in_sum_s_993: Item = Item(672, 0.0, 209346562.0, mutable=False, type="int")
    bytes_in_min_s_993: Item = Item(673, 0.0, 42232512.0, mutable=False, type="int")
    bytes_in_max_s_993: Item = Item(674, 0.0, 209011253.0, mutable=False, type="int")
    bytes_out_sum_s_993: Item = Item(675, 0.0, 6822169.0, mutable=True, type="int")
    bytes_out_min_s_993: Item = Item(676, 0.0, 650114.0, mutable=True, type="int")
    bytes_out_max_s_993: Item = Item(677, 0.0, 6816438.0, mutable=True, type="int")
    pkts_in_sum_s_993: Item = Item(678, 0.0, 156108.0, mutable=False, type="int")
    pkts_in_min_s_993: Item = Item(679, 0.0, 32165.0, mutable=False, type="int")
    pkts_in_max_s_993: Item = Item(680, 0.0, 154716.0, mutable=False, type="int")
    pkts_out_sum_s_993: Item = Item(681, 0.0, 77336.0, mutable=True, type="int")
    pkts_out_min_s_993: Item = Item(682, 0.0, 15984.0, mutable=True, type="int")
    pkts_out_max_s_993: Item = Item(683, 0.0, 75503.0, mutable=True, type="int")
    duration_sum_s_993: Item = Item(684, 0.0, 42410.0, mutable=True, type="real")
    duration_min_s_993: Item = Item(685, 0.0, 21961.0, mutable=True, type="real")
    duration_max_s_993: Item = Item(686, 0.0, 21961.0, mutable=True, type="real")
    tcp_sum_s_993: Item = Item(687, 0.0, 209485697.0, mutable=True, type="int")
    udp_sum_s_993: Item = Item(688, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_s_993: Item = Item(689, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_s_993: Item = Item(690, 0.0, 4.0, mutable=False, type="int")
    distinct_src_port_s_993: Item = Item(691, 0.0, 16.0, mutable=False, type="int")
    distinct_dst_port_s_993: Item = Item(692, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_d_993: Item = Item(693, 0.0, 1939065.0, mutable=False, type="int")
    bytes_in_min_d_993: Item = Item(694, 0.0, 1939065.0, mutable=False, type="int")
    bytes_in_max_d_993: Item = Item(695, 0.0, 1939065.0, mutable=False, type="int")
    bytes_out_sum_d_993: Item = Item(696, 0.0, 215081.0, mutable=True, type="int")
    bytes_out_min_d_993: Item = Item(697, 0.0, 72807.0, mutable=True, type="int")
    bytes_out_max_d_993: Item = Item(698, 0.0, 137322.0, mutable=True, type="int")
    pkts_in_sum_d_993: Item = Item(699, 0.0, 6072.0, mutable=False, type="int")
    pkts_in_min_d_993: Item = Item(700, 0.0, 1897.0, mutable=False, type="int")
    pkts_in_max_d_993: Item = Item(701, 0.0, 1897.0, mutable=False, type="int")
    pkts_out_sum_d_993: Item = Item(702, 0.0, 6232.0, mutable=True, type="int")
    pkts_out_min_d_993: Item = Item(703, 0.0, 1949.0, mutable=True, type="int")
    pkts_out_max_d_993: Item = Item(704, 0.0, 1949.0, mutable=True, type="int")
    duration_sum_d_993: Item = Item(705, 0.0, 74576.0, mutable=True, type="real")
    duration_min_d_993: Item = Item(706, 0.0, 21992.0, mutable=True, type="real")
    duration_max_d_993: Item = Item(707, 0.0, 21998.0, mutable=True, type="real")
    tcp_sum_d_993: Item = Item(708, 0.0, 1942544.0, mutable=True, type="int")
    udp_sum_d_993: Item = Item(709, 0.0, 0.0, mutable=True, type="int")
    icmp_sum_d_993: Item = Item(710, 0.0, 0.0, mutable=True, type="int")
    distinct_external_ips_d_993: Item = Item(711, 0.0, 2.0, mutable=False, type="int")
    distinct_src_port_d_993: Item = Item(712, 0.0, 11.0, mutable=False, type="int")
    distinct_dst_port_d_993: Item = Item(713, 0.0, 1.0, mutable=False, type="int")
    bytes_in_sum_s_OTHER: Item = Item(714, 0.0, 3913263415.0, mutable=False, type="int")
    bytes_in_min_s_OTHER: Item = Item(715, 0.0, 360897755.0, mutable=False, type="int")
    bytes_in_max_s_OTHER: Item = Item(716, 0.0, 1774132850.0, mutable=False, type="int")
    bytes_out_sum_s_OTHER: Item = Item(717, 0.0, 5101358572.0, mutable=True, type="int")
    bytes_out_min_s_OTHER: Item = Item(718, 0.0, 12223614.0, mutable=True, type="int")
    bytes_out_max_s_OTHER: Item = Item(719, 0.0, 1844175440.0, mutable=True, type="int")
    pkts_in_sum_s_OTHER: Item = Item(720, 0.0, 2837011.0, mutable=False, type="int")
    pkts_in_min_s_OTHER: Item = Item(721, 0.0, 277039.0, mutable=False, type="int")
    pkts_in_max_s_OTHER: Item = Item(722, 0.0, 1227489.0, mutable=False, type="int")
    pkts_out_sum_s_OTHER: Item = Item(723, 0.0, 1567499.0, mutable=True, type="int")
    pkts_out_min_s_OTHER: Item = Item(724, 0.0, 201856.0, mutable=True, type="int")
    pkts_out_max_s_OTHER: Item = Item(725, 0.0, 1287818.0, mutable=True, type="int")
    duration_sum_s_OTHER: Item = Item(726, 0.0, 809976.0, mutable=True, type="real")
    duration_min_s_OTHER: Item = Item(727, 0.0, 22030.0, mutable=True, type="real")
    duration_max_s_OTHER: Item = Item(728, 0.0, 22030.0, mutable=True, type="real")
    tcp_sum_s_OTHER: Item = Item(729, 0.0, 5101361215.0, mutable=True, type="int")
    udp_sum_s_OTHER: Item = Item(730, 0.0, 997523610.0, mutable=True, type="int")
    icmp_sum_s_OTHER: Item = Item(731, 0.0, 12376.0, mutable=True, type="int")
    distinct_external_ips_s_OTHER: Item = Item(
        732,
        0.0,
        1370.0,
        mutable=False,
        type="int",
    )
    distinct_src_port_s_OTHER: Item = Item(733, 0.0, 201.0, mutable=False, type="int")
    distinct_dst_port_s_OTHER: Item = Item(734, 0.0, 1361.0, mutable=False, type="int")
    bytes_in_sum_d_OTHER: Item = Item(
        735,
        0.0,
        11630589400.0,
        mutable=False,
        type="int",
    )
    bytes_in_min_d_OTHER: Item = Item(
        736,
        0.0,
        11630589400.0,
        mutable=False,
        type="int",
    )
    bytes_in_max_d_OTHER: Item = Item(
        737,
        0.0,
        11630589400.0,
        mutable=False,
        type="int",
    )
    bytes_out_sum_d_OTHER: Item = Item(738, 0.0, 6045566530.0, mutable=True, type="int")
    bytes_out_min_d_OTHER: Item = Item(739, 0.0, 191012435.0, mutable=True, type="int")
    bytes_out_max_d_OTHER: Item = Item(740, 0.0, 6045487440.0, mutable=True, type="int")
    pkts_in_sum_d_OTHER: Item = Item(741, 0.0, 8050083.0, mutable=False, type="int")
    pkts_in_min_d_OTHER: Item = Item(742, 0.0, 8050083.0, mutable=False, type="int")
    pkts_in_max_d_OTHER: Item = Item(743, 0.0, 8050083.0, mutable=False, type="int")
    pkts_out_sum_d_OTHER: Item = Item(744, 0.0, 4155952.0, mutable=True, type="int")
    pkts_out_min_d_OTHER: Item = Item(745, 0.0, 4155952.0, mutable=True, type="int")
    pkts_out_max_d_OTHER: Item = Item(746, 0.0, 4155952.0, mutable=True, type="int")
    duration_sum_d_OTHER: Item = Item(747, 0.0, 1312704.0, mutable=True, type="real")
    duration_min_d_OTHER: Item = Item(748, 0.0, 22034.0, mutable=True, type="real")
    duration_max_d_OTHER: Item = Item(749, 0.0, 22034.0, mutable=True, type="real")
    tcp_sum_d_OTHER: Item = Item(750, 0.0, 11630589400.0, mutable=True, type="int")
    udp_sum_d_OTHER: Item = Item(751, 0.0, 208769368.0, mutable=True, type="int")
    icmp_sum_d_OTHER: Item = Item(752, 0.0, 86828.0, mutable=True, type="int")
    distinct_external_ips_d_OTHER: Item = Item(
        753,
        0.0,
        2985.0,
        mutable=False,
        type="int",
    )
    distinct_src_port_d_OTHER: Item = Item(754, 0.0, 16378.0, mutable=False, type="int")
    distinct_dst_port_d_OTHER: Item = Item(755, 0.0, 58606.0, mutable=False, type="int")
    window_timestamp: Item = Item(
        756,
        1312962390,
        1313593920,
        mutable=False,
        type="int",
    )


ctu_meta = CtuMeta()
ctu_meta_dict = {k: Item(**v) for (k, v) in asdict(ctu_meta).items()}

categorical_index = [
    item.idx for item in ctu_meta_dict.values() if item.type in ("int", "cat")
]
immutable_index = [item.idx for item in ctu_meta_dict.values() if not item.mutable]


# relational constraints

# Avoid looking up the index on every check
g1_left_idx = [
    item.idx
    for name, item in ctu_meta_dict.items()
    if name.startswith(("icmp_sum_s_", "udp_sum_s_", "tcp_sum_s_"))
]

g1_right_idx = [
    item.idx
    for name, item in ctu_meta_dict.items()
    if name.startswith(("bytes_in_sum_s_", "bytes_out_sum_s_"))
]


def g1(x: Point) -> bool:
    """Check that the per-protocol sums for source ports equal the in/out byte sums for source ports (g1).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    left_sum = sum(x[idx] for idx in g1_left_idx)
    right_sum = sum(x[idx] for idx in g1_right_idx)
    if not (abs(left_sum - right_sum) < EPS):
        logger.debug(f"[g1] sum mismatch: {left_sum} != {right_sum}")
        return False
    return True


g2_left_idx: Final = [
    item.idx
    for name, item in ctu_meta_dict.items()
    if name.startswith(("icmp_sum_d_", "udp_sum_d_", "tcp_sum_d_"))
]
g2_right_idx: Final = [
    item.idx
    for name, item in ctu_meta_dict.items()
    if name.startswith(("bytes_in_sum_d_", "bytes_out_sum_d_"))
]


def g2(x: Point) -> bool:
    """Check that the per-protocol sums for destination ports equal the in/out byte sums for destination ports (g2).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    left_sum = sum(x[idx] for idx in g2_left_idx)
    right_sum = sum(x[idx] for idx in g2_right_idx)

    if not (abs(left_sum - right_sum) < EPS):
        logger.debug(f"[g2] sum mismatch: {left_sum} != {right_sum}")
        return False
    return True


packet_size = []
for e in ["s", "d"]:
    for port in [
        "1",
        "3",
        "8",
        "10",
        "21",
        "22",
        "25",
        "53",
        "80",
        "110",
        "123",
        "135",
        "138",
        "161",
        "443",
        "445",
        "993",
    ]:
        bytes_idx = ctu_meta_dict[f"bytes_out_sum_{e}_{port}"].idx
        pkts_idx = ctu_meta_dict[f"pkts_out_sum_{e}_{port}"].idx
        packet_size.append((bytes_idx, pkts_idx))


min_max_sum_triplets = []
for e_1 in ["bytes_out", "pkts_out", "duration"]:
    for port in [
        "1",
        "3",
        "8",
        "10",
        "21",
        "22",
        "25",
        "53",
        "80",
        "110",
        "123",
        "135",
        "138",
        "161",
        "443",
        "445",
        "993",
        "OTHER",
    ]:
        for e_2 in ["d", "s"]:
            max_idx = ctu_meta_dict[f"{e_1}_max_{e_2}_{port}"].idx
            min_idx = ctu_meta_dict[f"{e_1}_min_{e_2}_{port}"].idx
            sum_idx = ctu_meta_dict[f"{e_1}_sum_{e_2}_{port}"].idx
            min_max_sum_triplets.append(
                (max_idx, min_idx, sum_idx),
            )


def g_type(x: Point) -> bool:
    """Check that all categorical features of the CTU dataset take integer values.

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for idx in categorical_index:
        round_value = round(x[idx])
        if abs(x[idx] - round_value) > EPS:
            logger.debug(f"[g_type] x[{idx}] must be integer: got {x[idx]}")
            return False
    return True


def g_min_max(x: Point) -> bool:
    """Check that every feature of the CTU dataset lies within its [min, max] range.

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for col_name, item in ctu_meta_dict.items():
        if item.min is None or item.max is None:
            continue
        if not (item.min <= x[item.idx] <= item.max):
            logger.debug(
                f"[g_min_max] {col_name} value {x[item.idx]} out of bounds "
                f"[{item.min}, {item.max}]",
            )
            return False
    return True


def g_immutable(victim_x: Point, x: Point) -> bool:
    """Check that all immutable features of the CTU dataset are unchanged.

    Args:
        victim_x: Original point being attacked.
        x: Modified point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for item in ctu_meta_dict.values():
        if item.mutable:
            continue
        if abs(x[item.idx] - victim_x[item.idx]) > EPS:
            return False
    return True


def bound_box(victim_x: Point) -> BoundingBox:
    """Build the bounding box of the CTU dataset around the victim point.

    Mutable features are bounded by their [min, max] range and immutable features are fixed to the victim's value.

    Args:
        victim_x: Original point being attacked.

    Returns:
        Bounding box in which adversarial points can exist.

    """
    lower = [-float("inf")] * victim_x.shape[0]
    upper = [float("inf")] * victim_x.shape[0]
    is_set = [False] * victim_x.shape[0]
    for item in ctu_meta_dict.values():
        if item.mutable:
            assert item.min is not None
            assert item.max is not None
            lower[item.idx] = item.min
            upper[item.idx] = item.max
        else:
            lower[item.idx] = victim_x[item.idx]
            upper[item.idx] = victim_x[item.idx]
        is_set[item.idx] = True

    # Prevent forgetting to set the ranges
    assert all(is_set)
    return BoundingBox(np.array(lower), np.array(upper))


def create_sum(indices: list[int]) -> Feature | Constant | MathOperation:
    """Build a Moeva2 expression for the sum of the given features.

    Args:
        indices: Indices of the features to sum.

    Returns:
        Sum expression, or the constant 0 if indices is empty.

    """
    if not indices:
        return Constant(0)
    res = Feature(indices[0])
    for idx in indices[1:]:
        res = res + Feature(idx)
    return res


class CtuCostraints(BaseConstraints):
    """Constraints for the CTU dataset."""

    def __init__(self, victim_x: Point) -> None:
        """Initialize the constraints for the given victim point.

        Args:
            victim_x: Original point being attacked.

        """
        super().__init__(victim_x)
        self.relationship_consts: list[Callable[[Point], bool]] = [
            g_min_max,
            lambda x: g_immutable(victim_x, x),
            g_type,
            g1,
            g2,
        ]

        for _bytes_idx, _pkts_idx in packet_size:
            # Default arguments are used because of the late-binding closure issue, see
            # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
            def g(
                x: Point,
                bytes_idx: int = _bytes_idx,
                pkts_idx: int = _pkts_idx,
            ) -> bool:
                """Check that the average packet size (bytes / packets) does not exceed 1500.

                Args:
                    x: Point to check.
                    bytes_idx: Index of the bytes feature.
                    pkts_idx: Index of the packets feature.

                Returns:
                    True if the constraint is satisfied, False otherwise.

                """
                if not (
                    abs(x[pkts_idx]) < EPS or (x[bytes_idx] / x[pkts_idx]) <= 1500 + EPS
                ):
                    return False
                    logger.debug("g_packet failed")
                return True

            self.relationship_consts.append(g)

        for _max_idx, _min_idx, _sum_idx in min_max_sum_triplets:

            def g(
                x: Point,
                max_idx: int = _max_idx,
                min_idx: int = _min_idx,
                sum_idx: int = _sum_idx,
            ) -> bool:
                """Check that min <= max <= sum, and min <= sum, for one triplet of features.

                Args:
                    x: Point to check.
                    max_idx: Index of the max feature.
                    min_idx: Index of the min feature.
                    sum_idx: Index of the sum feature.

                Returns:
                    True if the constraint is satisfied, False otherwise.

                """
                if not (x[max_idx] <= x[sum_idx] + EPS):
                    logger.debug("g-max-sum failed")
                    return False
                if not (x[min_idx] <= x[sum_idx] + EPS):
                    logger.debug(
                        f"g-min-sum failed: {x[min_idx]} <= {x[sum_idx]}"
                        " ({min_idx = } {sum_idx = })",
                    )
                    return False
                if not (x[min_idx] <= x[max_idx] + EPS):
                    logger.debug("g-min-max failed")
                    return False
                return True

            self.relationship_consts.append(g)

        self.bound_box = bound_box(victim_x)

    @override
    def get_categorial_index(self) -> list[int]:
        """Return the indices of discrete features of the CTU dataset.

        Returns:
            List of feature indices.

        """
        return categorical_index

    @override
    def get_immutable_index(self) -> list[int]:
        """Return the indices of immutable features of the CTU dataset.

        Returns:
            List of feature indices.

        """
        return immutable_index

    @override
    def get_bound_box(self) -> BoundingBox:
        """Return the region where adversarial points can exist for the CTU dataset.

        Returns:
            Bounding box of the search region.

        """
        return self.bound_box

    @override
    def is_valid(self, x: Point) -> bool:
        """Check if x satisfies all constraints of the CTU dataset.

        Args:
            x: Point to check.

        Returns:
            True if the constraint is satisfied, False otherwise.

        """
        return all(func(x) for func in self.relationship_consts)

    @override
    def set_smt_constraints(self, s: Solver, x: list[ArithRef]) -> None:
        # g1
        """Add the relational constraints of the CTU dataset to the SMT solver.

        Args:
            s: Z3 solver to add the constraints to.
            x: Z3 variables corresponding to the features.

        Returns:
            None.

        """
        s.add(Sum(x[xi] for xi in g1_left_idx) == Sum(x[xi] for xi in g1_right_idx))
        # g2
        s.add(Sum(x[xi] for xi in g2_left_idx) == Sum(x[xi] for xi in g2_right_idx))
        # g_packet_size
        for bytes_idx, pkts_idx in packet_size:
            s.add(
                Or(
                    [
                        And(x[pkts_idx] == 0),
                        x[bytes_idx] <= 1500 * x[pkts_idx],
                    ],
                ),
            )
        # g_min_max_sum
        for max_idx, min_idx, sum_idx in min_max_sum_triplets:
            s.add(x[max_idx] <= x[sum_idx])
            s.add(x[min_idx] <= x[sum_idx])
            s.add(x[min_idx] <= x[max_idx])

    @override
    def set_scip_constraints(self, s: Model, x: list[Variable]) -> None:
        # g1
        """Add the relational constraints of the CTU dataset to the SCIP model.

        Args:
            s: SCIP model to add the constraints to.
            x: SCIP variables corresponding to the features.

        Returns:
            None.

        """
        s.addCons(
            sum(x[xi] for xi in g1_left_idx) == sum(x[xi] for xi in g1_right_idx),
        )
        # g2
        s.addCons(
            sum(x[xi] for xi in g2_left_idx) == sum(x[xi] for xi in g2_right_idx),
        )
        # g_packet_size
        for (
            bytes_idx,
            pkts_idx,
        ) in packet_size:
            s.addCons(x[bytes_idx] <= 1500 * x[pkts_idx])
        # g_min_max_sum
        for max_idx, min_idx, sum_idx in min_max_sum_triplets:
            s.addCons(x[max_idx] <= x[sum_idx])
            s.addCons(x[min_idx] <= x[sum_idx])
            s.addCons(x[min_idx] <= x[max_idx])

    @override
    def get_moeva_constraints(self) -> list[BaseRelationConstraint]:
        """Return the relational constraints of the CTU dataset for Moeva2.

        Returns:
            List of relation constraints.

        """
        relation_constraints = []

        # g1
        g1_left = create_sum(g1_left_idx)
        g1_right = create_sum(g1_right_idx)
        if g1_left_idx or g1_right_idx:
            relation_constraints.append(g1_left == g1_right)

        # g2
        g2_left = create_sum(g2_left_idx)
        g2_right = create_sum(g2_right_idx)
        if g2_left_idx or g2_right_idx:
            relation_constraints.append(g2_left == g2_right)

        # g_packet_size
        for bytes_idx, pkts_idx in packet_size:
            relation_constraints.append(
                Feature(bytes_idx)
                <= (Feature(pkts_idx) * Constant(1500)) + Constant(1e-4),
            )

        # g_min_max_sum
        for (
            max_idx,
            min_idx,
            sum_idx,
        ) in min_max_sum_triplets:
            relation_constraints.append(
                Feature(max_idx) <= Feature(sum_idx) + Constant(1e-4),
            )
            relation_constraints.append(
                Feature(min_idx) <= Feature(sum_idx) + Constant(1e-4),
            )
            relation_constraints.append(
                Feature(min_idx) <= Feature(max_idx) + Constant(1e-4),
            )

        return relation_constraints
