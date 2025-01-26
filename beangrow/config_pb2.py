# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0c\x63onfig.proto\x12\x11\x62\x65\x61ncount.returns\"\xad\x01\n\x06\x43onfig\x12\x38\n\x0binvestments\x18\x01 \x01(\x0b\x32#.beancount.returns.InvestmentConfig\x12.\n\x06groups\x18\x03 \x01(\x0b\x32\x1e.beancount.returns.GroupConfig\x12\x39\n\x13\x62\x65nchmark_portfolio\x18\x02 \x03(\x0b\x32\x1c.beancount.returns.Portfolio\"u\n\x10InvestmentConfig\x12\x31\n\ninvestment\x18\x01 \x03(\x0b\x32\x1d.beancount.returns.Investment\x12\x15\n\rincome_regexp\x18\x02 \x01(\t\x12\x17\n\x0f\x65xpenses_regexp\x18\x03 \x01(\t\"\x7f\n\nInvestment\x12\x10\n\x08\x63urrency\x18\x01 \x01(\t\x12\x15\n\rasset_account\x18\x02 \x01(\t\x12\x19\n\x11\x64ividend_accounts\x18\x03 \x03(\t\x12\x16\n\x0ematch_accounts\x18\x04 \x03(\t\x12\x15\n\rcash_accounts\x18\x05 \x03(\t\"i\n\x0bGroupConfig\x12\'\n\x05group\x18\x01 \x03(\x0b\x32\x18.beancount.returns.Group\x12\x31\n\nsupergroup\x18\x02 \x03(\x0b\x32\x1d.beancount.returns.SuperGroup\"X\n\x05Group\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\ninvestment\x18\x02 \x03(\t\x12\x10\n\x08\x63urrency\x18\x03 \x01(\t\x12\x1b\n\x13\x62\x65nchmark_portfolio\x18\x04 \x03(\t\")\n\nSuperGroup\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05group\x18\x02 \x03(\t\"0\n\x08Position\x12\x12\n\ninstrument\x18\x01 \x01(\t\x12\x10\n\x08\x66raction\x18\x02 \x01(\x02\"H\n\tPortfolio\x12\x0c\n\x04name\x18\x01 \x01(\t\x12-\n\x08position\x18\x02 \x03(\x0b\x32\x1b.beancount.returns.Position')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'config_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CONFIG._serialized_start=36
  _CONFIG._serialized_end=209
  _INVESTMENTCONFIG._serialized_start=211
  _INVESTMENTCONFIG._serialized_end=328
  _INVESTMENT._serialized_start=330
  _INVESTMENT._serialized_end=457
  _GROUPCONFIG._serialized_start=459
  _GROUPCONFIG._serialized_end=564
  _GROUP._serialized_start=566
  _GROUP._serialized_end=654
  _SUPERGROUP._serialized_start=656
  _SUPERGROUP._serialized_end=697
  _POSITION._serialized_start=699
  _POSITION._serialized_end=747
  _PORTFOLIO._serialized_start=749
  _PORTFOLIO._serialized_end=821
# @@protoc_insertion_point(module_scope)
