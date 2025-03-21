#!/usr/bin/env python3
import os
import glob
import sys
import argparse
import sqlite3
import traceback
from datetime import datetime
from inspect import getframeinfo, stack
from typing import List, Any, Optional, Dict, Callable, Union

# Minimalistic Logger для вывода сообщений
class Logger:
    @staticmethod
    def info(msg: str):
        print(f"[INFO] {msg}")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {msg}", file=sys.stderr)

    @staticmethod
    def error_exit(msg: str):
        print(f"[ERROR] {msg}", file=sys.stderr)
        sys.exit(1)

# Класс для хранения аргументов (замена argparse.Namespace)
class Dummy:
    input: Optional[str] = None
    readme: bool = False
    auto_inc_achs: bool = False
    rem_dup_ops: bool = False
    rem_all_ops: bool = False

    # пакет
    app: Optional[str] = None
    app_id: Optional[str] = None

    # игрок
    player: Optional[str] = None

    # листинг
    list_cc: bool = False
    list_games: bool = False
    list_players: bool = False
    list_ops: bool = False

    # листинг ачивок
    list_achs: bool = False
    list_u_achs: bool = False
    list_nu_achs: bool = False
    list_nor_achs: bool = False
    list_inc_achs: bool = False
    list_sec_achs: bool = False

    # поиск
    search_games: Optional[str] = None
    search_achs: Optional[str] = None
    search_u_achs: Optional[str] = None
    search_nu_achs: Optional[str] = None
    search_nor_achs: Optional[str] = None
    search_inc_achs: Optional[str] = None
    search_sec_achs: Optional[str] = None

    # разблокировка
    unlock_id: Optional[str] = None
    unlock_all: bool = False
    unlock_listed: bool = False

# Базовый класс для всех сущностей
class Wrapper:
    _id: Optional[int] = None
    _changers: Dict[str, Callable] = {}
    _args: List[Any] = []
    _hidden_attrs: List[str] = [
        "_changers",
        "_args",
        "_hidden_attrs"
    ]

    def __init__(self, *args) -> None:
        self._args = args

    def attrs(self) -> List[str]:
        return [x for x in self.__dict__ if not x.startswith("__") and not x.endswith("__") and x not in self._hidden_attrs]

    def values(self) -> List[Any]:
        return [getattr(self, x) for x in self.attrs()]

    def dict(self) -> Dict[str, Any]:
        return {x: getattr(self, x) for x in self.attrs()}

    def dump(self, changers=None) -> None:
        caller = getframeinfo(stack()[1][0])
        fname, line = caller.filename, caller.lineno
        fname = os.path.relpath(fname, os.getcwd())
        inst = self.apply_changers(changers)
        vals = {x: getattr(inst, x) for x in inst.attrs()}
        dmp = "\n\t".join(f"{x} = {y}" for x, y in vals.items())
        name = self.__class__.__name__
        print(f"{fname}:{line}:{name}:\n\t{dmp}")

    def get_arg(self, index: int, lst: List[Any]) -> Any:
        return lst[index] if index < len(lst) else None

    @staticmethod
    def join(*args, sep=" : ") -> str:
        return sep.join(str(x) for x in args)

    @property
    def id(self) -> Optional[int]:
        return self._id

    def apply_changers(self, changers=None) -> "Wrapper":
        if changers is None:
            changers = list(self._changers.keys())
        d = self.dict()
        for n, f in self._changers.items():
            if n in d and n in changers:
                d[n] = f(d[n])
        return self.__class__(*list(d.values()))

    def __repr__(self) -> str:
        ps = getattr(self, "print_string", None)
        nm = self.__class__.__name__
        return f"<{nm} '{ps() if ps else self.id}'>"

# Определение достижения
class AchievementDefinition(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.game_id = self.get_arg(1, args)
        self.external_achievement_id = self.get_arg(2, args)
        self.type = self.get_arg(3, args)
        self.name = self.get_arg(4, args)
        self.description = self.get_arg(5, args)
        self.unlocked_icon_image_id = self.get_arg(6, args)
        self.revealed_icon_image_id = self.get_arg(7, args)
        self.total_steps = self.get_arg(8, args)
        self.formatted_total_steps = self.get_arg(9, args)
        self.initial_state = self.get_arg(10, args)
        self.sorting_rank = self.get_arg(11, args)
        self.definition_xp_value = self.get_arg(12, args)
        self.rarity_percent = self.get_arg(13, args)

    def is_normal(self):
        return self.type == 0

    def is_incremental(self):
        return self.type == 1

    def is_secret(self):
        return self.initial_state == 2

    def print_string(self, npad=0, dpad=0):
        typ = "[INC" if self.is_incremental() else "[NOR"
        typ += "|SEC]" if self.is_secret() else "]"
        name = self.name
        desc = self.description
        if npad: name = name[:npad] + "..." if len(name) > npad else name
        if dpad: desc = desc[:dpad] + "..." if len(desc) > dpad else desc
        return self.join(typ, self.external_achievement_id, name, desc, f"{self.definition_xp_value}xp")

# Экземпляр достижения
class AchievementInstance(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._changers = {
            "last_updated_timestamp": lambda x: datetime.fromtimestamp(x / 1000)
        }
        self._id = self.get_arg(0, args)
        self.definition_id = self.get_arg(1, args)
        self.player_id = self.get_arg(2, args)
        self.state = self.get_arg(3, args)
        self.current_steps = self.get_arg(4, args)
        self.formatted_current_steps = self.get_arg(5, args)
        self.last_updated_timestamp = self.get_arg(6, args)
        self.instance_xp_value = self.get_arg(7, args)

    def is_unlocked(self) -> bool:
        return self.state == 0

    def is_locked(self) -> bool:
        return self.state > 0

    def print_string(self):
        return self.join(self.definition_id, self.state, self.current_steps, self.instance_xp_value)

# Операция для достижения
class AchievementPendingOp(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.client_context_id = self.get_arg(1, args)
        self.external_achievement_id = self.get_arg(2, args)
        self.achievement_type = self.get_arg(3, args)
        self.new_state = self.get_arg(4, args)
        self.steps_to_increment = self.get_arg(5, args)
        self.min_steps_to_set = self.get_arg(6, args)
        self.external_game_id = self.get_arg(7, args)
        self.external_player_id = self.get_arg(8, args)

    def print_string(self):
        return self.join(self.client_context_id, self.external_achievement_id, self.external_game_id, self.external_player_id)

# Клиентский контекст
class ClientContext(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._changers = {
            "account_name": lambda x: x[0:2] + "*" * (len(x) - x.find("@") - 2) + x[x.find("@")-2:]
        }
        self._id = self.get_arg(0, args)
        self.package_name = self.get_arg(1, args)
        self.package_uid = self.get_arg(2, args)
        self.account_name = self.get_arg(3, args)
        self.account_type = self.get_arg(4, args)
        self.is_games_lite = self.get_arg(5, args)

    def print_string(self):
        return self.join(self.id, self.package_name)

# Информация об экземпляре игры
class GameInstance(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.instance_game_id = self.get_arg(1, args)
        self.real_time_support = self.get_arg(2, args)
        self.turn_based_support = self.get_arg(3, args)
        self.platform_type = self.get_arg(4, args)
        self.instance_display_name = self.get_arg(5, args)
        self.package_name = self.get_arg(6, args)
        self.piracy_check = self.get_arg(7, args)
        self.installed = self.get_arg(8, args)
        self.preferred = self.get_arg(9, args)
        self.gamepad_support = self.get_arg(10, args)

# Идентификатор игрока в игре
class GamePlayerId(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.game_player_ids_external_player_id = self.get_arg(1, args)
        self.game_player_ids_external_game_id = self.get_arg(2, args)
        self.game_player_ids_external_game_player_id = self.get_arg(3, args)
        self.game_player_ids_external_primary_player_id = self.get_arg(4, args)
        self.game_player_ids_created_in_epoch = self.get_arg(5, args)

# Информация об игре
class Game(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.external_game_id = self.get_arg(1, args)
        self.display_name = self.get_arg(2, args)
        self.primary_category = self.get_arg(3, args)
        self.secondary_category = self.get_arg(4, args)
        self.developer_name = self.get_arg(5, args)
        self.game_description = self.get_arg(6, args)
        self.game_icon_image_id = self.get_arg(7, args)
        self.game_hi_res_image_id = self.get_arg(8, args)
        self.featured_image_id = self.get_arg(9, args)
        self.screenshot_image_ids = self.get_arg(10, args)
        self.screenshot_image_widths = self.get_arg(11, args)
        self.screenshot_image_heights = self.get_arg(12, args)
        self.video_url = self.get_arg(13, args)
        self.play_enabled_game = self.get_arg(14, args)
        self.last_played_server_time = self.get_arg(15, args)
        self.last_connection_local_time = self.get_arg(16, args)
        self.last_synced_local_time = self.get_arg(17, args)
        self.metadata_version = self.get_arg(18, args)
        self.sync_token = self.get_arg(19, args)
        self.metadata_sync_requested = self.get_arg(20, args)
        self.target_instance = self.get_arg(21, args)
        self.gameplay_acl_status = self.get_arg(22, args)
        self.availability = self.get_arg(23, args)
        self.owned = self.get_arg(24, args)
        self.achievement_total_count = self.get_arg(25, args)
        self.leaderboard_count = self.get_arg(26, args)
        self.price_micros = self.get_arg(27, args)
        self.formatted_price = self.get_arg(28, args)
        self.full_price_micros = self.get_arg(29, args)
        self.formatted_full_price = self.get_arg(30, args)
        self.explanation = self.get_arg(31, args)
        self.description_snippet = self.get_arg(32, args)
        self.starRating = self.get_arg(33, args)
        self.ratingsCount = self.get_arg(34, args)
        self.muted = self.get_arg(35, args)
        self.identity_sharing_confirmed = self.get_arg(36, args)
        self.snapshots_enabled = self.get_arg(37, args)
        self.theme_color = self.get_arg(38, args)
        self.lastUpdatedTimestampMillis = self.get_arg(39, args)

    def print_string(self, inst: Optional[GameInstance]=None):
        middle = self.developer_name if inst is None else inst.package_name
        return self.join(self.external_game_id, middle, self.display_name)

# Изображение
class Image(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.url = self.get_arg(1, args)
        self.local = self.get_arg(2, args)
        self.filesize = self.get_arg(3, args)
        self.download_timestamp = self.get_arg(4, args)

# Игрок
class Player(Wrapper):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._id = self.get_arg(0, args)
        self.external_player_id = self.get_arg(1, args)
        self.profile_name = self.get_arg(2, args)
        self.profile_icon_image_id = self.get_arg(3, args)
        self.profile_hi_res_image_id = self.get_arg(4, args)
        self.last_updated = self.get_arg(5, args)
        self.is_in_circles = self.get_arg(6, args)
        self.current_xp_total = self.get_arg(7, args)
        self.current_level = self.get_arg(8, args)
        self.current_level_min_xp = self.get_arg(9, args)
        self.current_level_max_xp = self.get_arg(10, args)
        self.next_level = self.get_arg(11, args)
        self.next_level_max_xp = self.get_arg(12, args)
        self.last_level_up_timestamp = self.get_arg(13, args)
        self.player_title = self.get_arg(14, args)
        self.has_all_public_acls = self.get_arg(15, args)
        self.has_debug_access = self.get_arg(16, args)
        self.is_profile_visible = self.get_arg(17, args)
        self.most_recent_activity_timestamp = self.get_arg(18, args)
        self.most_recent_external_game_id = self.get_arg(19, args)
        self.most_recent_game_name = self.get_arg(20, args)
        self.most_recent_game_icon_id = self.get_arg(21, args)
        self.most_recent_game_hi_res_id = self.get_arg(22, args)
        self.most_recent_game_featured_id = self.get_arg(23, args)
        self.gamer_tag = self.get_arg(24, args)
        self.real_name = self.get_arg(25, args)
        self.banner_image_landscape_id = self.get_arg(26, args)
        self.banner_image_portrait_id = self.get_arg(27, args)
        self.total_unlocked_achievements = self.get_arg(28, args)
        self.play_together_friend_status = self.get_arg(29, args)
        self.play_together_nickname = self.get_arg(30, args)
        self.play_together_invitation_nickname = self.get_arg(31, args)
        self.profile_creation_timestamp = self.get_arg(32, args)
        self.nickname_abuse_report_token = self.get_arg(33, args)
        self.friends_list_visibility = self.get_arg(34, args)
        self.always_auto_sign_in = self.get_arg(35, args)

# Finder для поиска объектов в базе данных
class Finder:
    def __init__(self, db_instance: "DbFile") -> None:
        self.db: DbFile = db_instance

    def ach_def_by_id(self, x: Any) -> Optional[AchievementDefinition]:
        return self.db.select_by_cls_fe(AchievementDefinition, ["_id"], [x])

    def ach_def_by_external_id(self, x: Any) -> Optional[AchievementDefinition]:
        return self.db.select_by_cls_fe(AchievementDefinition, ["external_achievement_id"], [x])

    def ach_inst_by_id(self, x: Any) -> Optional[AchievementInstance]:
        return self.db.select_by_cls_fe(AchievementInstance, ["_id"], [x])

    def client_context_by_id(self, x: Any) -> Optional[ClientContext]:
        return self.db.select_by_cls_fe(ClientContext, ["_id"], [x])

    def game_inst_by_id(self, x: Any) -> Optional[GameInstance]:
        return self.db.select_by_cls_fe(GameInstance, ["_id"], [x])

    def game_player_id_by_id(self, x: Any) -> Optional[GamePlayerId]:
        return self.db.select_by_cls_fe(GamePlayerId, ["_id"], [x])

    def game_by_id(self, x: Any) -> Optional[Game]:
        return self.db.select_by_cls_fe(Game, ["_id"], [x])

    def image_by_id(self, x: Any) -> Optional[Image]:
        return self.db.select_by_cls_fe(Image, ["_id"], [x])

    def ach_def_by_ach_inst(self, x: AchievementInstance) -> Optional[AchievementDefinition]:
        return self.db.select_by_cls_fe(AchievementDefinition, ["_id"], [x.definition_id])

    def ach_defs_by_ach_insts(self, x: List[AchievementInstance]) -> List[Optional[AchievementDefinition]]:
        return [self.ach_def_by_ach_inst(y) for y in x]

    def game_inst_by_game(self, x: Game) -> Optional[GameInstance]:
        return self.db.select_by_cls_fe(GameInstance, ["instance_game_id", "installed"], [x.id, 1])

    def game_insts_by_games(self, x: List[Game]) -> List[Optional[GameInstance]]:
        return [self.game_inst_by_game(y) for y in x]

    def game_inst_by_game_id(self, x: Any) -> Optional[GameInstance]:
        return self.db.select_by_cls_fe(GameInstance, ["instance_game_id"], [x])

    def game_inst_by_package_name(self, x: str) -> Optional[GameInstance]:
        return self.db.select_by_cls_fe(GameInstance, ["package_name"], [x])

    def game_by_game_inst(self, x: GameInstance) -> Optional[Game]:
        return self.db.select_by_cls_fe(Game, ["_id"], [x.instance_game_id])

    def game_by_external_id(self, x: Any) -> Optional[Game]:
        return self.db.select_by_cls_fe(Game, ["external_game_id"], [x])

    def game_by_ach_inst(self, x: AchievementInstance) -> Optional[Game]:
        udef = self.ach_def_by_ach_inst(x)
        if udef:
            return self.db.select_by_cls_fe(Game, ["_id"], [udef.game_id])
        return None

    def game_by_name(self, search: Any) -> Optional[Game]:
        return self.db.search_by_cls_fe(search, cls=Game)

    def games_by_name(self, search: Any) -> List[Game]:
        return self.db.search_by_cls(search, cls=Game)

    def game_by_ach_def(self, x: AchievementDefinition) -> Optional[Game]:
        return self.db.select_by_cls_fe(Game, ["_id"], [x.game_id])

    def games_by_ach_defs(self, x: List[AchievementDefinition]) -> List[Optional[Game]]:
        return [self.game_by_ach_def(y) for y in x]

    def ach_defs_by_game_id(self, x: Any) -> List[Optional[AchievementDefinition]]:
        return self.db.select_by_cls(AchievementDefinition, ["game_id"], [x], exact=True)

    def ach_defs_by_game(self, x: Game) -> List[Optional[AchievementDefinition]]:
        return self.ach_defs_by_game_id(x.id)

    def ach_inst_by_ach_def(self, x: AchievementDefinition) -> Optional[AchievementInstance]:
        return self.db.select_by_cls_fe(AchievementInstance, ["definition_id"], [x.id])

    def ach_insts_by_ach_defs(self, x: List[AchievementDefinition]) -> List[Optional[AchievementInstance]]:
        return [self.ach_inst_by_ach_def(y) for y in x]

    def ach_insts_by_game_id(self, x: Any) -> List[Optional[AchievementInstance]]:
        ugame = self.game_by_id(x)
        if ugame:
            udefs = self.ach_defs_by_game(ugame)
            if udefs:
                return self.ach_insts_by_ach_defs(udefs)
        return []

    def ach_insts_by_game(self, x: Game) -> List[Optional[AchievementInstance]]:
        return self.ach_insts_by_game_id(x.id)

    def client_context_by_game_inst(self, x: GameInstance) -> Optional[ClientContext]:
        return self.db.select_by_cls_fe(ClientContext, ["package_name"], [x.package_name])

# Класс для работы с базой данных
class DbFile:
    mapping = {
        "achievement_pending_ops": AchievementPendingOp,
        "achievement_definitions": AchievementDefinition,
        "achievement_instances": AchievementInstance,
        "client_contexts": ClientContext,
        "game_instances": GameInstance,
        "game_player_ids": GamePlayerId,
        "games": Game,
        "images": Image,
        "players": Player
    }

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.cur = self.connection.cursor()

    def __get_table_by_cls(self, cls: type):
        try:
            cls_index = list(self.mapping.values()).index(cls)
            return list(self.mapping.keys())[cls_index]
        except ValueError:
            name = cls.__name__ if isinstance(cls, type) else cls
            Logger.error_exit(f"Given class '{name}' doesn't have associated table")

    def select_by_cls(self, cls: type=None, cols: List[str]=None, values: List[Any]=None, first: bool=False, exact: bool=False):
        return self.select(cls=cls, cols=cols, values=values, first=first, exact=exact)

    def select_by_cls_fe(self, cls: type=None, cols: List[str]=None, values: List[Any]=None):
        return self.select(cls=cls, cols=cols, values=values, first=True, exact=True)

    def search_by_cls(self, search: str, cls: type=None, first: bool=False, exact: bool=False):
        return self.search(search=search, cls=cls, first=first, exact=exact)

    def search_by_cls_fe(self, search: str, cls: type=None):
        return self.search(search=search, cls=cls, first=True, exact=True)

    def search(self, search: str, table: str=None, cls: type=None, first: bool=False, exact: bool=False):
        if cls is not None:
            table = self.__get_table_by_cls(cls)
        s = str(search).lower()
        sql = f"select * from {table} where (1=0"
        for col in cls().attrs():
            if exact:
                sql += f" or lower({col})='{s}'"
            else:
                sql += f" or lower({col}) like '%{s}%'"
        sql += ") order by _id"
        res = self.cur.execute(sql).fetchall()
        res = [x if cls is None else cls(*x) for x in res]
        if len(res):
            return res[0] if first else res
        return None if first else []

    @staticmethod
    def search_instances(search: Any, objs: List[Wrapper], first: bool=False, exact: bool=False):
        s = str(search).lower()
        res = list(filter(
            lambda x: any(s == str(y).lower() if exact else s in str(y).lower() for y in x.values()), objs))
        if len(res):
            return res[0] if first else res
        return None if first else []

    def search_instances_by(self, s: Any, cols: List[str]=None, objs: List[Wrapper]=None, first: bool=False, exact: bool=False):
        if not objs:
            return []
        if not cols:
            return self.search_instances(s, objs, first)
        res = []
        s = str(s).lower()
        for obj in objs:
            attrs = obj.attrs()
            attrs = [x for x in cols if x in attrs]
            values = [str(getattr(obj, x)).lower() for x in attrs]
            if any(s == x if exact else s in x for x in values):
                res.append(obj)
        if len(res):
            return res[0] if first else res
        return None if first else []

    def select(self, table: str=None, cols: List[str]=None, values: List[Any]=None, cls: type=None, first: bool=False, exact: bool=False):
        if cls is not None:
            table = self.__get_table_by_cls(cls)
        cil = isinstance(cols, list)
        if not cil: cols = []
        vil = isinstance(values, list)
        if not vil: values = []
        if cil and vil and len(cols) != len(values):
            Logger.error_exit("Given cols doesn't have appropriate number of values")
        sql = f"select * from {table} where 1=1"
        for c, v in zip(cols, values):
            if exact:
                sql += f" and {c}='{v}'"
            else:
                sql += f" and {c} like '%{v}%'"
        sql += " order by _id"
        res = self.cur.execute(sql).fetchall()
        res = [x if cls is None else cls(*x) for x in res]
        if len(res):
            return res[0] if first else res
        return None if first else []

    def ex(self, table: str):
        return self.cur.execute("select * from " + table + " order by _id")

    def remove_duplicate_pending_ops(self, by_col: str="external_achievement_id"):
        ops = [AchievementPendingOp(*x) for x in self.ex("achievement_pending_ops").fetchall()]
        seen = set()
        removed = 0
        for op in ops:
            if getattr(op, by_col) in seen:
                self.cur.execute("delete from achievement_pending_ops where _id = ?", (op.id,))
                removed += 1
            else:
                seen.add(getattr(op, by_col))
        self.connection.commit()
        return removed

    def empty_pending_ops(self):
        self.cur.execute("delete from achievement_pending_ops")
        self.connection.commit()

    def add_pending_op(self, op: Dict[str, Any]):
        sql = "insert into achievement_pending_ops values ({})".format(
            ",".join("?" for _ in range(len(op))))
        self.cur.execute(sql, list(op.values()))
        self.connection.commit()

    def get_next_pending_op_id(self):
        res = self.cur.execute("select max(_id) from achievement_pending_ops").fetchone()[0]
        return 0 if res is None else res + 1

# Основной класс для работы с ачивками Google Play
class GooglePlayAchievementUnlocker:
    default_db_regex = "/data/data/com.google.android.gms/databases/games_*.db"
    # default_db_regex = "dbs/games_*.db"

    readme_text = """
HOW TO USE?
1) Disconnect from the internet
2) Unlock the achievements you want
3) Reconnect to the internet
4) Run Google Play Games to sync the achievements
5) Profit

ACHIEVEMENT FLAGS?
NOR - normal
INC - incremental
SEC - secret

GAME WON'T APPEAR IN --list-cc? Try one of these:
1) Play the game for a couple of minutes
2) In-app button to logout and login again
3) Earn any achievement
4) Re/Open Google Play App
5) Clear Cache/All data and login again
6) Restart phone
"""

    def __init__(self, a):
        self.args: Dummy = a
        self.inst_db: Optional[DbFile] = None
        self.inst_finder: Optional[Finder] = None
        self.reload()

    @property
    def db(self) -> DbFile:
        assert self.inst_db is not None, "Database not loaded"
        return self.inst_db

    @property
    def finder(self) -> Finder:
        assert self.inst_finder is not None, "Finder not loaded"
        return self.inst_finder

    def get_db_files(self):
        return glob.glob(self.default_db_regex)

    def reload(self):
        self.check_player()
        if self.args.input is None:
            files = self.get_db_files()
            if len(files) == 0:
                Logger.error_exit("No database file found")
            self.args.input = files[0]
        errors = self.load_db_file(self.args.input)
        if errors:
            Logger.error_exit("\n".join(errors))

    def load_db_file(self, file=None):
        if file is None:
            file = self.get_db_files()[0]
        errors = []
        if not os.path.isfile(file):
            errors.append("Input is not a file")
        if not os.access(file, os.R_OK):
            errors.append("Input is not readable")
        try:
            self.inst_db = DbFile(sqlite3.connect(file))
            self.inst_finder = Finder(self.db)
        except Exception:
            errors.append(traceback.format_exc())
        return errors

    def run(self):
        if self.args.readme:
            Logger.error_exit(self.readme_text)
        try:
            if self.args.rem_all_ops:
                Logger.info("Removing all pending achievement ops...")
                self.db.empty_pending_ops()
            achs: List[AchievementDefinition] = []
            if self.args.list_cc:
                ccs = self.db.select(cls=ClientContext)
                print("\n".join([x.print_string() for x in ccs]))
            elif self.args.list_games:
                games = self.db.select(cls=Game)
                game_insts = [self.finder.game_inst_by_game(x) for x in games]
                print("\n".join([g.print_string(gi) for g, gi in zip(games, game_insts) if gi]))
            elif self.args.list_players:
                self.list_players()
            elif self.args.list_ops:
                ops: List[AchievementPendingOp] = [x for x in self.db.select(cls=AchievementPendingOp) if x]
                print("\n".join([x.print_string() for x in ops]))
            elif getattr(self.args, "scan_all", False):
                # Новая команда: сканирование всех игр и вывод достижений
                games = self.db.select(cls=Game)
                for game in games:
                    print(f"Game: {game.display_name} (ID: {game.external_game_id})")
                    ach_defs = self.finder.ach_defs_by_game(game)
                    if ach_defs:
                        for ach in ach_defs:
                            print("  " + ach.print_string())
                    else:
                        print("  No achievements found for this game.")
            if self.args.list_achs:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_defs = [x for x in self.finder.ach_defs_by_game_id(app.id) if x]
                achs = ach_defs
            elif self.args.list_u_achs:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_insts = [x for x in self.finder.ach_insts_by_game_id(app.id) if x and x.is_unlocked()]
                achs = [x for x in self.finder.ach_defs_by_ach_insts(ach_insts) if x]
            elif self.args.list_nu_achs:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_insts = [x for x in self.finder.ach_insts_by_game_id(app.id) if x and x.is_locked()]
                achs = [x for x in self.finder.ach_defs_by_ach_insts(ach_insts) if x]
            elif self.args.list_nor_achs:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_defs = [x for x in self.finder.ach_defs_by_game_id(app.id) if x and x.is_normal()]
                achs = ach_defs
            elif self.args.list_inc_achs:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_defs = [x for x in self.finder.ach_defs_by_game_id(app.id) if x and x.is_incremental()]
                achs = ach_defs
            elif self.args.list_sec_achs:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_defs = [x for x in self.finder.ach_defs_by_game_id(app.id) if x and x.is_secret()]
                achs = ach_defs
            opt_app = self.get_app(optional=True)
            if self.args.search_games:
                search = self.args.search_games
                found_games: List[Game] = self.db.search(search, cls=Game)
                print("\n".join([x.print_string(self.finder.game_inst_by_game(x)) for x in found_games]))
            elif self.args.search_achs:
                achs = self.find_achievements(self.args.search_achs, opt_app)
            elif self.args.search_u_achs:
                achs = self.find_achievements(self.args.search_u_achs, opt_app, locked=False)
            elif self.args.search_nu_achs:
                achs = self.find_achievements(self.args.search_nu_achs, opt_app, unlocked=False)
            elif self.args.search_nor_achs:
                achs = self.find_achievements(self.args.search_nor_achs, opt_app, inc=False, sec=False)
            elif self.args.search_inc_achs:
                achs = self.find_achievements(self.args.search_inc_achs, opt_app, nor=False, sec=False)
            elif self.args.search_sec_achs:
                achs = self.find_achievements(self.args.search_sec_achs, opt_app, nor=False, inc=False)
            if len(achs):
                print("\n".join([x.print_string() for x in achs]))
            if self.args.unlock_id:
                self.unlock_achievement(self.finder.ach_def_by_external_id(self.args.unlock_id))
            elif self.args.unlock_all:
                app = self.get_app()
                assert app is not None, "Package not found"
                ach_insts = [x for x in self.finder.ach_insts_by_game_id(app.id) if x]
                for ach_inst in ach_insts:
                    self.unlock_achievement(self.finder.ach_def_by_ach_inst(ach_inst))
            elif self.args.unlock_listed:
                for ach in achs:
                    self.unlock_achievement(ach)
            if self.args.rem_dup_ops:
                Logger.info("Removing duplicate pending achievement ops...")
                removed = self.db.remove_duplicate_pending_ops()
                Logger.info(f"Removed: {removed}")
        except Exception:
            Logger.error_exit(f"{traceback.format_exc()}\nSomething bad has happened, probably a bug or uncut edge case.\nPlease report this to the developer.")

    def unlock_achievement(self, ach_def: Optional[AchievementDefinition]=None):
        if ach_def is None:
            return
        ach_inst = self.finder.ach_inst_by_ach_def(ach_def)
        if ach_inst is None:
            Logger.error("Achievement definition doesn't have an associated achievement instance")
            return
        if ach_inst.is_unlocked():
            Logger.info(f"Achievement {ach_def.external_achievement_id} is already unlocked...")
            return
        game = self.finder.game_by_ach_inst(ach_inst)
        assert game is not None, "Game not found"
        game_inst = None if not game else self.finder.game_inst_by_game(game)
        client_context = None if not game_inst else self.finder.client_context_by_game_inst(game_inst)
        if not client_context:
            Logger.error("No client context found for this game")
            return
        package_name = "NO_INSTANCE" if not game_inst else game_inst.package_name
        Logger.info(f"Unlocking achievement {ach_def.external_achievement_id} ({package_name})...")
        steps_to_increment = ""
        if ach_def.is_incremental():
            if self.args.auto_inc_achs:
                steps_to_increment = ach_def.total_steps - ach_inst.current_steps
            else:
                print(f"### {ach_def.name} | {ach_def.description} | {ach_def.definition_xp_value}xp")
                print(f"### Progress: {ach_inst.current_steps}/{ach_def.total_steps}")
                steps_to_increment = self.get_increment_value()
        self.db.add_pending_op({
            "_id": self.db.get_next_pending_op_id(),
            "client_context_id": client_context.id,
            "external_achievement_id": ach_def.external_achievement_id,
            "achievement_type": ach_def.type,
            "new_state": 0,
            "steps_to_increment": steps_to_increment,
            "min_steps_to_set": "",
            "external_game_id": game.external_game_id,
            "external_player_id": self.get_player_id(),
        })

    def get_increment_value(self):
        print(end='\r')
        value = input("### Steps to increment by: ")
        try:
            if int(value) < 0:
                raise ValueError
            return value
        except ValueError:
            Logger.error("Value must be a number bigger than -1")
            return self.get_increment_value()

    def find_achievements(self, search, package=None, unlocked=True, locked=True, nor=True, inc=True, sec=True) -> List[AchievementDefinition]:
        achs = self.finder.ach_defs_by_game(package) if package else self.db.select(cls=AchievementDefinition)
        found_achs: List[AchievementDefinition] = self.db.search_instances_by(search, ["name", "description"], achs)
        ach_insts = self.finder.ach_insts_by_ach_defs(found_achs)
        ach_defs_filtered = [
            x for x, w in zip(found_achs, ach_insts)
            if (w and w.is_unlocked() and unlocked) or (w and w.is_locked() and locked)
        ]
        ach_defs_filtered = [
            x for x in ach_defs_filtered
            if (x.is_normal() and nor) or (x.is_incremental() and inc) or (x.is_secret() and sec)
        ]
        return ach_defs_filtered

    def get_player_id(self):
        players: List[Player] = self.db.select(cls=Player)
        if not players:
            Logger.error_exit("Couldn't find any player")
        return players[0].external_player_id

    def list_players(self):
        players = self.get_players()
        index = 1
        for db_file, db_player in players.items():
            file_name = os.path.basename(db_file)
            print(f"[{index}] ({file_name}) {db_player.print_string()}")
            index += 1

    def get_players(self) -> Dict[str, Player]:
        current_db_file = self.args.input
        players = {}
        for db_file in self.get_db_files():
            errors = self.load_db_file(db_file)
            if not errors:
                db_players = self.db.select(cls=Player)
                if db_players:
                    players[db_file] = db_players[0]
        self.load_db_file(current_db_file)
        return players

    def check_player(self):
        player = self.args.player
        if player is None:
            return
        db_players = self.get_players()
        try:
            ip = int(player)
            if ip < 1 or ip > len(db_players):
                raise ValueError
            self.args.input = list(db_players.keys())[ip - 1]
        except ValueError:
            Logger.error_exit(f"Player must be a number from 1 to {len(db_players)}")

    def get_app(self, optional=False) -> Optional[Game]:
        if self.args.app is None and self.args.app_id is None:
            if optional:
                return None
            Logger.error_exit("No package specified (use -a or -aid)")
        if self.args.app is not None:
            app_inst = self.finder.game_inst_by_package_name(self.args.app)
            if not app_inst:
                if optional:
                    return None
                Logger.error_exit("Package not found")
            if app_inst:
                return self.finder.game_by_game_inst(app_inst)
        app = self.finder.game_by_external_id(self.args.app_id)
        if not app:
            if optional:
                return None
            Logger.error_exit("Package not found")
        return app

# Точка входа
if __name__ == "__main__":
    DEBUG = False
    args: Union[argparse.Namespace, Dummy] = Dummy()
    parser = argparse.ArgumentParser(epilog="By @TheNoiselessNoise")
    if not DEBUG:
        parser.add_argument('-i', dest='input', metavar='input', help='Path to the .db file')
        parser.add_argument('--readme', dest='readme', action='store_true', help='How to use this?')
        parser.add_argument('--auto-inc-achs', dest='auto_inc_achs', action='store_true', help='Automatically set incremental achievements to max')
        parser.add_argument('--rem-dup-ops', dest='rem_dup_ops', action='store_true', help='Remove duplicate achievement pending ops')
        parser.add_argument('--rem-all-ops', dest='rem_all_ops', action='store_true', help='Remove all achievement pending ops')
        package_group = parser.add_mutually_exclusive_group()
        package_group.add_argument('-a', dest='app', metavar='app_name', help='App name')
        package_group.add_argument('-aid', dest='app_id', metavar='app_id', help='App id')
        player_group = parser.add_mutually_exclusive_group()
        player_group.add_argument('-p', dest='player', metavar='#', help='Player number in --list-players')
        list_group = parser.add_mutually_exclusive_group()
        list_group.add_argument('--list-cc', action='store_true', help='List all client contexts')
        list_group.add_argument('--list-games', action='store_true', help='List all games')
        list_group.add_argument('--list-players', action='store_true', help='List all players')
        list_group.add_argument('--list-ops', action='store_true', help='List all achievement pending ops')
        package_list_group = parser.add_mutually_exclusive_group()
        package_list_group.add_argument('--list-achs', action='store_true', help='List all achievements')
        package_list_group.add_argument('--list-u-achs', action='store_true', help='List all unlocked achievements')
        package_list_group.add_argument('--list-nu-achs', action='store_true', help='List all not unlocked achievements')
        package_list_group.add_argument('--list-nor-achs', action='store_true', help='List all normal achievements')
        package_list_group.add_argument('--list-inc-achs', action='store_true', help='List all incremental achievements')
        package_list_group.add_argument('--list-sec-achs', action='store_true', help='List all secret achievements')
        # Новая команда для автоматического сканирования всех игр и вывода их ачивок
        parser.add_argument('--scan-all', action='store_true', help='Scan all games and output all achievements automatically')
        search_group = parser.add_argument_group()
        search_group.add_argument('--search-games', metavar='search', type=str, help='Search for a game')
        search_group.add_argument('--search-achs', metavar='search', type=str, help='Search for achievements')
        search_group.add_argument('--search-u-achs', metavar='search', type=str, help='Search for unlocked achievements')
        search_group.add_argument('--search-nu-achs', metavar='search', type=str, help='Search for not unlocked achievements')
        search_group.add_argument('--search-nor-achs', metavar='search', type=str, help='Search for normal achievements')
        search_group.add_argument('--search-inc-achs', metavar='search', type=str, help='Search for incremental achievements')
        search_group.add_argument('--search-sec-achs', metavar='search', type=str, help='Search for secret achievements')
        unlock_group = parser.add_mutually_exclusive_group()
        unlock_group.add_argument('--unlock-id', dest='unlock_id', metavar='external_id', type=str, help='Unlock an achievement by its external id')
        unlock_group.add_argument('--unlock-all', dest='unlock_all', action='store_true', help='Unlock all achievements in the given package')
        unlock_group.add_argument('--unlock-listed', dest='unlock_listed', action='store_true', help='Unlock all listed achievements')
        args = parser.parse_args()
    else:
        args.input = "dbs\\games_2db19fbf.db"
    GooglePlayAchievementUnlocker(args).run()
