class Params:
	"""
	:ivar ahk_cmd: 
	:type ahk_cmd: str

	:ivar auth_file: 
	:type auth_file: str

	:ivar dst_path: 
	:type dst_path: str

	:ivar info_dir: 
	:type info_dir: str

	:ivar info_file: 
	:type info_file: str

	:ivar info_root: 
	:type info_root: str

	:ivar key_dir: 
	:type key_dir: str

	:ivar key_root: 
	:type key_root: str

	:ivar log_file: 
	:type log_file: str

	:ivar mode: 
	:type mode: int

	:ivar port: 
	:type port: str

	:ivar scp_dst: 
	:type scp_dst: str

	:ivar scp_name: 
	:type scp_name: str

	:ivar scp_path: 
	:type scp_path: str

	:ivar src_info: 
	:type src_info: str

	:ivar use_ahk: 
	:type use_ahk: int

	:ivar wait_t: 
	:type wait_t: int

	:ivar win_title: 
	:type win_title: str

	"""
	def __init__(self):
		self.cfg = ()
		self.ahk_cmd = 'paste_with_cat_1'
		self.auth_file = ''
		self.dst_path = '.'
		self.info_dir = ''
		self.info_file = ''
		self.info_root = ''
		self.key_dir = ''
		self.key_root = ''
		self.log_file = ''
		self.mode = 0
		self.port = ''
		self.scp_dst = ''
		self.scp_name = ''
		self.scp_path = '.'
		self.src_info = ''
		self.use_ahk = 1
		self.wait_t = 10
		self.win_title = 'The Journal 8'
