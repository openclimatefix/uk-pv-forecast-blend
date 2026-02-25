"""database interactions for forecasts."""
from sqlalchemy.orm.session import Session
from datetime import datetime
from nowcasting_datamodel.read.read import get_forecast_values_latest


def get_forecast_values_from_db(
    session: Session,
    gsp_id: int,
    model_name: str,
    start_datetime: datetime | None,
):
    """Get forecast values from the database (nowcasting_datamodel)."""
    return get_forecast_values_latest(
        session=session,
        gsp_id=gsp_id,
        start_datetime=start_datetime,
        model_name=model_name,
    )
