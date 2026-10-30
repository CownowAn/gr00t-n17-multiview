from multiview_tasks.tasks import (
    blind_insertion, catch_ball, color_button, door_fruit, door_hold, lit_button, push_drawer, ring_peg, roll_can, roll_can_dirs, roll_can_table, roll_ball_table, shelf_take, side_ball,
)

# Task name -> {level: robosuite env name}
TASK_ENV_NAMES = {
    blind_insertion.BlindInsertionBottleCabinet.TASK_NAME: blind_insertion.ENV_NAMES,
    color_button.ColorButtonBox.TASK_NAME: color_button.ENV_NAMES,
    color_button.ColorButtonBoxV2.TASK_NAME: color_button.ENV_NAMES_V2,
    color_button.ColorButtonBoxV3.TASK_NAME: color_button.ENV_NAMES_V3,
    push_drawer.PushDrawerBoxV3.TASK_NAME: push_drawer.ENV_NAMES,
    ring_peg.RingOnPeg.TASK_NAME: ring_peg.ENV_NAMES,
    door_hold.HoldDoorTakeBlock.TASK_NAME: door_hold.ENV_NAMES,
    door_hold.HoldDoorTakeBlockV2.TASK_NAME: door_hold.ENV_NAMES_V2,
    door_hold.HoldDoorTakeBlockV3.TASK_NAME: door_hold.ENV_NAMES_V3,
    door_fruit.HoldDoorTakeFruitV3.TASK_NAME: door_fruit.ENV_NAMES,
    shelf_take.ShelfTakeBlock.TASK_NAME: shelf_take.ENV_NAMES,
    catch_ball.CatchRollingBall.TASK_NAME: catch_ball.ENV_NAMES,
    side_ball.CatchSideBall.TASK_NAME: side_ball.ENV_NAMES,
    lit_button.PressLitButton.TASK_NAME: lit_button.ENV_NAMES,
    roll_can.StopRollingCan.TASK_NAME: roll_can.ENV_NAMES,
    roll_can_dirs.StopRollingCanDirs.TASK_NAME: roll_can_dirs.ENV_NAMES,
    roll_can_table.StopRollingCanTable.TASK_NAME: roll_can_table.ENV_NAMES,
    roll_ball_table.StopRollingBallTable.TASK_NAME: roll_ball_table.ENV_NAMES,
}
