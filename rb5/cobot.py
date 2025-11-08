from socket import *
import threading
import time
import struct
from dataclasses import dataclass
from enum import Enum
#from multipledispatch import dispatch
# from PyQt5.QtCore import pyqtSlot
#import logging
from loguru import logger

#import rainbow

@dataclass
class systemSTAT:
    time: float = None
    jnt_ref: tuple = None
    jnt_ang: tuple = None
    cur: tuple = None
    tcp_ref: tuple = None
    tcp_pos: tuple = None
    analog_in: tuple = None
    analog_out: tuple = None
    digital_in: tuple = None
    digital_out: tuple = None
    temperature_mc: tuple = None
    task_pc: int = None
    task_repeat: int = None
    task_run_id: int = None
    task_run_num: int = None
    task_run_time: float = None
    task_state: int = None
    default_speed: float = None
    robot_state: int = None
    power_state: int = None
    tcp_target: tuple = None
    jnt_info: tuple = None
    collision_detect_onoff: int = None
    is_freedrive_mode: int = None
    program_mode: int = None
    init_state_info: int = None
    init_error: int = None
    tfb_analog_in: tuple = None
    tfb_digital_in: tuple = None
    tfb_digital_out: tuple = None
    tfb_voltage_out: float = None
    op_stat_collision_occur: int = None
    op_stat_sos_flag: int = None
    op_stat_self_collision: int = None
    op_stat_soft_estop_occur: int = None
    op_stat_ems_flag: int = None
    digital_in_config: tuple = None
    inbox_trap_flag: tuple = None
    inbox_check_mode: tuple = None
    eft_fx: float = None
    eft_fy: float = None
    eft_fz: float = None
    eft_mx: float = None
    eft_my: float = None
    eft_mz: float = None


@dataclass
class Joint:
    j0: float = None
    j1: float = None
    j2: float = None
    j3: float = None
    j4: float = None
    j5: float = None


@dataclass
class Point:
    x: float = None
    y: float = None
    z: float = None
    rx: float = None
    ry: float = None
    rz: float = None


class COBOT_STATUS(Enum):
    IDLE = 0
    PAUSED = 1
    RUNNING = 2
    UNKNOWN = 3


class CMD_TYPE(Enum):
    MOVE = 0
    NONMOVE = 1


class PG_MODE(Enum):
    SIMULATION = 0
    REAL = 1


class CIRCLE_TYPE(Enum):
    INTENDED = 0
    CONSTANT = 1
    RADIAL = 2
    SMOOTH = 3


class CIRCLE_AXIS(Enum):
    X = 0
    Y = 1
    Z = 2


class BLEND_OPTION(Enum):
    RATIO = 0
    DISTANCE = 1


class BLEND_RTYPE(Enum):
    INTENDED = 0
    CONSTANT = 1

class BLEND_XB(Enum):
    SPEED = 0
    POSITION = 1

class ITPL_RTYPE(Enum):
    INTENDED = 0
    CONSTANT = 1
    RESERVED1 = 2
    SMOOTH = 3
    RESERVED2 = 4
    CA_INTENDED = 5
    CA_CONSTANT = 6
    RESERVED3 = 7
    CA_SMOOTH = 8


class SOS_FLAG(Enum):
    NONE = 0
    Encoder_ERR = 1                
    CPU_ERR = 2 
    BIG_ERR = 3 
    INPUT_ERR = 4 
    JAM_ERR = 5 
    OC_ERR = 6 
    POS_BOUND_ERR = 7 
    MODE_ERR = 8 
    MATCH_ERR = 9 
    OC_LV_ERR = 10
    TEMP_ERR = 11
    SPD_OVER_ERR = 12

class DOUT_SET(Enum):
    LOW = 0
    HIGH = 1
    BYPASS = 2


class DOUT_SET(Enum):
    VOLT_0 = 0
    VOLT_12 = 1
    VOLT_24 = 2


CMD_PORT = 5000
DATA_PORT = 5001
systemstat_global = systemSTAT()

cmd_connect = True
data_connect = True
bReadCmd = False
moveCmdFlag = False
# CMDSock = socket(AF_INET, SOCK_STREAM)
# DATASock = socket(AF_INET, SOCK_STREAM)
moveCmdCnt = 0
cmd_send_flag = 0  # read cmd에서 사용하기 위한 flag

__RB_VERSION__ = 'a-0.1'


def ConnectToCB(ip):
    try:
        # Check if the IP is valid
        if not isValidIP(ip):
            print('error')
            return False

        global CMDSock
        global DATASock
        CMDSock = socket(AF_INET, SOCK_STREAM)
        DATASock = socket(AF_INET, SOCK_STREAM)

        global cmd_connect
        global data_connect
        cmd_connect = CMDSock.connect_ex((ip, CMD_PORT))
        data_connect = DATASock.connect_ex((ip, DATA_PORT))

        if cmd_connect == 0 & data_connect == 0:
            CMDREAD_THREAD = threading.Thread(target=ReadCMD, args=(CMDSock,))
            DATAREAD_THREAD = threading.Thread(target=ReadDATA, args=(DATASock,))
            
            CMDREAD_THREAD.start()
            DATAREAD_THREAD.start()

            # pass
            return True
        
        else:
            logger.error("Connection Error for the robot")
            return False
            
    except Exception as e:
        print(f"Connection Error: {e}")
        raise e
        return False


def DisConnectToCB():
    global cmd_connect
    global data_connect
    cmd_connect = True
    data_connect = True

    time.sleep(1)

    CMDSock.close()
    DATASock.close()
    return True


def __Version():
    msg = 'RB-API : ' + __RB_VERSION__
    print(msg)


def GetCurrentJoint():
    if not data_connect:
        current_joint_ = Joint(systemstat_global.jnt_ref[0], systemstat_global.jnt_ref[1], systemstat_global.jnt_ref[2]
                               , systemstat_global.jnt_ref[3], systemstat_global.jnt_ref[4],
                               systemstat_global.jnt_ref[5])
        return current_joint_
    else:
        print("'data socket isn't connect")
        current_joint_ = Joint(0, 0, 0, 0, 0, 0)
        return current_joint_

def GetCurrentTCP():
    if not data_connect:
        current_tcp_ = Point(systemstat_global.tcp_ref[0], systemstat_global.tcp_ref[1],
                             systemstat_global.tcp_ref[2]
                             , systemstat_global.tcp_ref[3], systemstat_global.tcp_ref[4],
                             systemstat_global.tcp_ref[5])
        return current_tcp_
    else:
        print("'data socket isn't connect")
        current_tcp_ = Point(0, 0, 0, 0, 0, 0)
        return current_tcp_

def CobotInit():
    msg = 'mc jall init'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def SetProgramMode(mode=PG_MODE):
    msg = None
    if mode == PG_MODE.SIMULATION:
        msg = 'pgmode simulation'
    elif mode == PG_MODE.REAL:
        msg = 'pgmode real'

    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def SetRobotRealMode():
    msg = 'pgmode real'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def SetRobotSimulationMode():
    msg = 'pgmode simulation'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def MoveL(x, y, z, rx, ry, rz, spd, acc):
    msg = 'move_l(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(
        rz) + '], ' + str(spd) + ',' + str(acc) + ')'
    #print(msg)
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MoveJ(j0, j1, j2, j3, j4, j5, spd, acc):
    msg = 'move_j(jnt[' + str(j0) + ',' + str(j1) + ',' + str(j2) + ',' + str(j3) + ',' + str(j4) + ',' + str(
        j5) + '], ' + str(spd) + ',' + str(acc) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MoveJL(x, y, z, rx, ry, rz, spd, acc):
    msg = 'move_jl(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(
        rz) + '], ' + str(spd) + ',' + str(acc) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MoveJB_Clear():
    msg = 'move_jb_clear()'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MoveJB_Add(j0, j1, j2, j3, j4, j5):
    msg = 'move_jb_add(jnt[' + str(j0) + ',' + str(j1) + ',' + str(j2) + ',' + str(j3) + ',' + str(j4) + ',' + str(
        j5) + '])'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MoveJB_Run(spd=float, acc=float):
    msg = 'move_jb_run(' + str(spd) + ',' + str(acc) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MovePB_Clear():
    msg = 'move_pb_clear()'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MovePB_Add(x, y, z, rx, ry, rz, spd, option, quantity):
    b_option = 0
    if option == BLEND_OPTION.DISTANCE:

        b_option = 1
        if quantity < 0:
            quantity = 0
    elif option == BLEND_OPTION.RATIO:
        b_option = 0
        if quantity < 0:
            quantity = 0
        elif quantity > 1:
            quantity = 1

    msg = 'move_pb_add(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(
        rz) + '], ' + str(spd) + ',' + str(b_option) + ',' + str(quantity) + ')'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MovePB_Run(acc=float, type=BLEND_RTYPE):
    rtype = 0
    if type == BLEND_RTYPE.INTENDED:
        rtype = 0
    elif type == BLEND_RTYPE.CONSTANT:
        rtype = 1

    msg = 'move_pb_run(' + str(acc) + ',' + str(rtype) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MoveITPL_Clear():
    msg = 'move_itpl_clear()'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MoveITPL_Add(x, y, z, rx, ry, rz, spd):
    msg = 'move_itpl_add(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(
        rz) + '], ' + str(spd) + ')'

    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MoveITPL_Run(acc=float, type=ITPL_RTYPE):
    rtype = 0
    if type == ITPL_RTYPE.INTENDED:
        rtype = 0
    elif type == ITPL_RTYPE.CONSTANT:
        rtype = 1
    elif type == ITPL_RTYPE.SMOOTH:
        rtype = 3
    elif type == ITPL_RTYPE.CA_INTENDED:
        rtype = 5
    elif type == ITPL_RTYPE.CA_CONSTANT:
        rtype = 6
    elif type == ITPL_RTYPE.CA_SMOOTH:
        rtype = 8

    msg = 'move_itpl_run(' + str(acc) + ',' + str(rtype) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


# MoveXB implementation starts here
def MoveXB_Clear():
    msg = 'move_xb_clear()'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MoveXB_Add(spd, acc, option, b_val, pnt = [], jnt = []):
    b_option = 0
    if option == BLEND_OPTION.DISTANCE:
        b_option = 1
        if b_val < 0:
            b_val = 0
    elif option == BLEND_OPTION.RATIO:
        b_option = 0
        if b_val < 0:
            b_val = 0
        elif b_val > 1:
            b_val = 1
    
    if len(pnt) == 0 and len(jnt) == 0:
        print('pnt and jnt is empty or invalid')
        return False
    elif len(pnt) == 6 and len(jnt) == 0:
        x,y,z,rx,ry,rz = pnt
        msg = 'move_xb_add(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(
            rz) + '], ' + str(spd) + ',' + str(acc) + ',' + str(b_option) + ',' + str(b_val) + ')'

    elif len(pnt) == 0 and len(jnt) == 6:
        j0,j1,j2,j3,j4,j5 = jnt
        msg = 'move_xb_add(jnt[' + str(j0) + ',' + str(j1) + ',' + str(j2) + ',' + str(j3) + ',' + str(j4) + ',' + str(
            j5) + '], ' + str(spd) + ',' + str(acc) + ',' + str(b_option) + ',' + str(b_val) + ')'
            
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)



def MoveXB_Run(tb_opt=BLEND_XB):
    rtype = 0
    if tb_opt == BLEND_XB.SPEED:
        rtype = 0
    elif tb_opt == BLEND_XB.POSITION:
        rtype = 1

    msg = 'move_xb_run(' + str(rtype) + ',0)'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


# MoveXB implementation end here

def MoveLB_Clear():
    msg = 'move_lb_clear()'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def MoveLB_Add(x, y, z, rx, ry, rz, b_val):
    msg = 'move_lb_add(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(
        rz) + '], ' + str(b_val) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)

def MoveLB_Run(spd=float, acc=float, type=BLEND_RTYPE):
    rtype = 0
    if type == BLEND_RTYPE.INTENDED:
        rtype = 0
    elif type == BLEND_RTYPE.CONSTANT:
        rtype = 1

    msg = 'move_lb_run(' + str(spd) + ',' + str(acc) + ',' + str(rtype) + ')'
    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MoveCircle_ThreePoint(x1, y1, z1, rx1, ry1, rz1, x2, y2, z2, rx2, ry2, rz2, spd, acc, type):
    r_option = 0
    if type == CIRCLE_TYPE.INTENDED:
        r_option = 0
    elif type == CIRCLE_TYPE.CONSTANT:
        r_option = 1
    elif type == CIRCLE_TYPE.RADIAL:
        r_option = 2
    elif type == CIRCLE_TYPE.SMOOTH:
        r_option = 3

    msg = 'move_c_points(pnt[' + str(x1) + ',' + str(y1) + ',' + str(z1) + ',' + str(rx1) + ',' + str(
        ry1) + ',' + str(rz1) + '], pnt[' + str(x2) + ',' + str(y2) + ',' + str(z2) + ',' + str(
        rx2) + ',' + str(ry2) + ',' + str(rz2) + '], ' + str(spd) + ',' + str(acc) + ',' + str(r_option) + ')'

    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def MoveCircle_Axis(x, y, z, rx, ry, rz, axis, direction, angle, spd, acc, type):
    r_option = 0
    a_option = 0
    if type == CIRCLE_TYPE.INTENDED:
        r_option = 0
    elif type == CIRCLE_TYPE.CONSTANT:
        r_option = 1
    elif type == CIRCLE_TYPE.RADIAL:
        r_option = 2

    if axis == CIRCLE_AXIS.X:
        if direction == 1:
            a_option = '1,0,0'
        elif direction == -1:
            a_option = '-1,0,0'
        else:
            a_option = '1,0,0'
    elif axis == CIRCLE_AXIS.Y:
        if direction == 1:
            a_option = '0,1,0'
        elif direction == -1:
            a_option = '0,-1,0'
        else:
            a_option = '0,1,0'
    elif axis == CIRCLE_AXIS.Z:
        if direction == 1:
            a_option = '0,0,1'
        elif direction == -1:
            a_option = '0,0,-1'
        else:
            a_option = '0,0,1'
    else:
        print('axis-error')

    msg = 'move_c_axis(pnt[' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(
        ry) + ',' + str(rz) + '], ' + str(a_option) + ',' + str(angle) + ',' + str(spd) + ',' + str(
        acc) + ',' + str(r_option) + ')'

    return SendCOMMAND(msg, CMD_TYPE.MOVE)


def CBDigitalOut(port=float, type=-1):
    dstatus = 0
    if type == 0:
        dstatus = 0
    elif type == 1:
        dstatus = 1
    elif type == 2:
        dstatus = -1

    msg = 'set_box_dout(' + str(port) + ',' + str(dstatus) + ')'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def GetDigitalIn():
    return systemstat_global.digital_in

def SetBaseSpeed(spd):
    if spd > 1.0:
        spd = 1.0
    elif spd < 0.:
        spd = 0.
    # print(spd)
    msg = 'set_speed_bar(' + str(spd) + ')'
    # print(systemstat_global.jnt_ref)
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def MotionHalt():
    msg = 'task stop'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def MotionPause():
    msg = 'task pause'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def MotionResume():
    msg = 'task resume_a'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def CollisionResume():
    msg = 'task resume_b'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def RobotPowerDown():
    msg = "arm_powerdown()"
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def SetCollisionThreshold(threshold):
    msg = 'set rb_collision_th ' + str(threshold)
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def Set_TCP_Point(x, y, z, rx, ry, rz):
    msg = 'set_tcp_info(' + str(x) + ',' + str(y) + ',' + str(z) + ',' + str(rx) + ',' + str(ry) + ',' + str(rz) + ')'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)


def ManualScript(ex_msg):
    return SendCOMMAND(ex_msg, CMD_TYPE.NONMOVE)

def ReadCMD(sock):
    while True:
        retries = 5
        for attempt in range(retries):
            try:
                time.sleep(0.01)
                global cmd_send_flag
                global bReadCmd
                global moveCmdFlag, moveCmdCnt

                if cmd_send_flag == 1:
                    # Set timeout for socket receive
                    sock.settimeout(10)
                    msg_recv = sock.recv(26)
                    if msg_recv.decode('utf-8') == 'The command was executed\n':
                        cmd_send_flag = 0
                        bReadCmd = True

                        if moveCmdFlag:
                            moveCmdCnt = 3
                            systemstat_global.robot_state = 3
                            moveCmdFlag = False

                        bReadCmd = False

                if cmd_connect == 1 & data_connect == 1:
                    break

                # If we get here, the operation succeeded
                break

            except socket.timeout:
                logger.error(f"Timeout on attempt {attempt + 1} of {retries}")
                if attempt == retries - 1:  # Last attempt
                    logger.error("Max retries reached, giving up")
                    return
                continue
            except Exception as e:
                logger.error(f"Error in ReadCMD on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:  # Last attempt
                    break
                continue
        break

def ReadDATA(sock):
    t=time.time()
    while True:
        try:
            # if time.time()-t > 5:
            #     raise TimeoutError("Timeout (5s) in ReadDATA")


            msg = 'reqdata'
            sock.send(msg.encode('utf-8'))

            msg_recv = sock.recv(580)

            if msg_recv[0] == 0x24:
                size = int((msg_recv[2] << 8) | msg_recv[1])
                #print(size)
                if size <= len(msg_recv) - 4:
                    if msg_recv[3] == 3:

                        msg_recv_split = msg_recv[4:512]
                        result = struct.unpack(
                            'fffffffffffffffffffffffffffffffffffffffiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiffffffiiiififiiffffffiiiiiiiiiiiffiiiifiiiiiiiiiiiffffff',
                            msg_recv_split)
                        
                        global systemstat_global
                        systemstat = systemSTAT(result[0], result[1:7], result[7:13], result[13:19], result[19:25],
                                                result[25:31], result[31:35], result[35:39], result[39:55], result[55:71]
                                                , result[71:77], result[77], result[78], result[79], result[80], result[81],
                                                result[82], result[83], result[84], result[85], result[86:92]
                                                , result[92:98], result[98], result[99], result[100], result[101],
                                                result[102],
                                                result[103:105], result[105:107], result[107:109], result[109]
                                                , result[110], result[111], result[112], result[113], result[114],
                                                result[115:117], result[117:119], result[119:121], result[121], result[122]
                                                , result[123], result[124], result[125], result[126])
                        systemstat_global = systemstat
                        
                        
                        
                    elif msg_recv[3] == 4:
                        print('config')
                    elif msg_recv[10] == 10:
                        print('popup')

            if cmd_connect == 1 & data_connect == 1:
                break
        except Exception as e:
            logger.exception(f"Error in ReadDATA: {e}")
            break

def SendCOMMAND(str, cmd_type):
    global cmd_send_flag
    global bReadCmd, moveCmdFlag
    str_space = str + ' '
    
    max_retries = 5
    timeout = 10
    
    for attempt in range(max_retries):
        try:
            if cmd_type == CMD_TYPE.MOVE:
                start_time = time.time()
                while True:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Timeout waiting while SendCOMMAND (attempt {attempt+1}/{max_retries})")
                        
                    time.sleep(0.03)
                    if IsIdle() & (bReadCmd == False):
                        CMDSock.send(str_space.encode('utf-8'))
                        moveCmdFlag = True
                        cmd_send_flag = 1
                        systemstat_global.robot_state = 3
                        return True
                    elif IsPause():
                        return False
            else:
                CMDSock.send(str_space.encode('utf-8'))
                cmd_send_flag = 1
                return True
                
        except Exception as e:
            logger.exception(f"Error in SendCOMMAND (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                raise TimeoutError(f"Timeout waiting while SendCOMMAND (attempt {attempt+1}/{max_retries}) {e}")
            time.sleep(1)

def IsIdle():
    return systemstat_global.robot_state == 1

def IsRunning():
    return systemstat_global.robot_state == 3

def IsPause():
    return systemstat_global.op_stat_soft_estop_occur == 1


def IsInitialized():
    if systemstat_global.init_state_info == 6:
        return True
    else:
        return False


def IsRobotReal():
    if systemstat_global.program_mode == 0:
        return True
    else:
        return False

def IsDoorOpen():
    return systemstat_global.op_stat_sos_flag
    # return systemstat_global.op_stat_soft_estop_occur
    


def IsExternalCollisionOccured():
    return systemstat_global.op_stat_collision_occur


def IsSelfCollisionOccured():
    return systemstat_global.op_stat_self_collision

def IsCommandSockConnect():
    if not cmd_connect:
        # print('connect commmand')
        return True
    else:
        return False


def IsDataSockConnect():
    if not data_connect:
        # print('connect data')
        return True
    else:
        return False


def isValidIP(ip):
    ipsplit = ip.split(".")
    if (int(ipsplit[0]) < 0) | (int(ipsplit[1]) < 0) | (int(ipsplit[2]) < 0) | (int(ipsplit[3]) < 0):
        return False
    elif (int(ipsplit[0]) > 255) | (int(ipsplit[1]) > 255) | (int(ipsplit[2]) > 255) | (int(ipsplit[3]) > 255):
        return False
    else:
        return True


def GetCurrentCobotStatus():
    if systemstat_global.op_stat_soft_estop_occur == 1:
        return COBOT_STATUS.PAUSED

    if systemstat_global.robot_state == 1:
        return COBOT_STATUS.IDLE
    elif systemstat_global.robot_state == 3:
        return COBOT_STATUS.RUNNING
    else:
        return COBOT_STATUS.UNKNOWN


def ToCB(ip):
    if data_connect == False & cmd_connect == False:
        DisConnectToCB()
        print(f'\033[1m\033[91m   Disconnected from Robot\033[0m')
    elif data_connect == True & cmd_connect == True:
        ConnectToCB(ip)
        print(f'\033[1m\033[92m   Connected to Robot\033[0m')


def disconnect():
    DisConnectToCB()
    return True


def set_inboxes(inbox0_dx, inbox0_dy, inbox0_dz, inbox0_x, inbox0_y, inbox0_z, inbox1_dx, inbox1_dy, inbox1_dz, inbox1_x, inbox1_y, inbox1_z):
    msg = 'config inbox ' + str(inbox0_dx) + ',' + str(inbox0_dy) + ',' + str(inbox0_dz) + ',' + str(inbox0_x) + ',' + str(inbox0_y) + ',' + str(inbox0_z) + ',' + str(inbox1_dx) + ',' + str(inbox1_dy) + ',' + str(inbox1_dz) + ',' + str(inbox1_x) + ',' + str(inbox1_y) + ',' + str(inbox1_z)
    ManualScript(msg)

def pulse_DIO(dio_num=8, period=0.1):
    msg = 'set_dout_unit_pulse_shot(' + str(dio_num) + ',0,0,' + str(period) + ',0)'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

send_trigger = pulse_DIO

def set_io_as_inbox_out(io_num=8, inbox_func=28):
    msg = 'config io_function out ' + str(io_num) + ',' + str(inbox_func)
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def distance_based_dio(dio_num=8, distance=100):
    msg = 'set rb_dout_based_distance ' + '1,' + str(distance)+ "," + str(dio_num) + "," + '0.02,0'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def distance_based_dio_off(dio_num=8):
    msg = 'set rb_dout_based_distance ' + '0,0,' + str(dio_num) + "," + '0.1,0'
    return SendCOMMAND(msg, CMD_TYPE.NONMOVE)

def toggle_DIO(combination, period=0.1):
    output = []

    for i in range(16):
        if i in combination:
            output.append('1')
        else:
            output.append('-1')


    output = ','.join(map(str, output))


    msg ='set_dout_signal_toggle(' + ''.join(output) +  ')'
    Set = SendCOMMAND(msg, CMD_TYPE.NONMOVE)
    time.sleep(period)
    Reset = SendCOMMAND(msg, CMD_TYPE.NONMOVE)
    return Reset