/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MiniOps.td                                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::mini::MiniDialect)
namespace mlir {
namespace mini {

MiniDialect::MiniDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MiniDialect>())
    
     {
  
  initialize();
}

MiniDialect::~MiniDialect() = default;

} // namespace mini
} // namespace mlir
