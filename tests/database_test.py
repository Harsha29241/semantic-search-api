from app.services.database_service import get_connection


def main():
    connection = get_connection()

    print("Database connection successful!")

    connection.close()


if __name__ == "__main__":
    main()