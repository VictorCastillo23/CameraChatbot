def buscar_procedimiento_emergencia(query):
    procedimientos = {
        "incendio": "En caso de incendio, evacúe el edificio siguiendo las señales de emergencia y llame al 911.",
        "terremoto": "Durante un terremoto, manténgase alejado de ventanas y protéjase bajo una mesa resistente."
    }

    return procedimientos.get(query.lower(), "No se encontró un procedimiento para esa emergencia.")

def buscar_horarios(query):
    horarios = {
        "entrada principal": "Abierto de 6:00 AM a 10:00 PM.",
        "estacionamiento": "Abierto las 24 horas."
    }

    return horarios.get(query.lower(), "No se encontraron los horarios disponibles para esta ubicación.")

