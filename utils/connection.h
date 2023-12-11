//
// Created by Mete Akgun on 04.07.20.
//

#ifndef PML_CONNECTION_H
#define PML_CONNECTION_H

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <arpa/inet.h>
#include <unistd.h> //close
#include <arpa/inet.h> //close
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros
#include <errno.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <thread>
using namespace std;

uint64_t bytes_sent = 0;
uint64_t bytes_received =0;
double receive_time = 0.0;
double send_time = 0.0;


void RTimers(){
    receive_time = 0.0;
    send_time = 0.0;
}
int OpenSocket(string address, uint16_t port){
    int server_fd, sock;
    struct sockaddr_in addr;
    int addrlen = sizeof(addr);
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0){
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))){
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(address.c_str());
    addr.sin_port = htons(port);
    if (::bind(server_fd, (struct sockaddr *)&addr, sizeof(addr))<0){
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0){
        perror("listen");
        exit(EXIT_FAILURE);
    }
    if ((sock = accept(server_fd, (struct sockaddr *)&addr, (socklen_t*)&addrlen))<0){
        perror("accept");
        exit(EXIT_FAILURE);
    }
    int flag =1;
    if (setsockopt(sock,IPPROTO_TCP,TCP_NODELAY,(char *)&flag,sizeof(int)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    close(server_fd);
    return sock;
}

int ConnectToSocket(string address, uint16_t port){
    int sock = 0;
    struct sockaddr_in serv_addr;
    if ((sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
        printf("\n Socket creation error \n");
        exit(-1);
    }

    int flag =1;
    if (setsockopt(sock,IPPROTO_TCP,TCP_NODELAY,(char *)&flag,sizeof(int)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, address.c_str(), &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        exit(-1);
    }

    while (connect(sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
        sleep(1);
    }
    return sock;
}

void ReceiveSingular(int socket, uint8_t* buffer, size_t sz){
    size_t left = sz;
    while (left > 0){
        int received = read(socket , &(buffer)[sz-left] , left);
        left -= received;
    }
}

void SendSingular(int socket, uint8_t* buffer, size_t sz){
    size_t left = sz;
    while (left > 0){
        int sent = write(socket ,  &(buffer)[sz - left]  , left);
        left -= sent;
    }
}

void Receive(int *socket, uint8_t* buffer, size_t sz){
    thread thr[SOCKET_NUMBER];
    int block_size = (int)ceil(sz * 1.0 / SOCKET_NUMBER);
    if (block_size == 0)
        block_size = sz;
    for (int i=0; i < SOCKET_NUMBER; i++){
        thr[i] = thread(ReceiveSingular, *(socket + i), buffer + (i * block_size), block_size);
    }
    for (int i=0; i < SOCKET_NUMBER; i++){
        thr[i].join();
    }
    bytes_received+=sz;
}

void Send(int *socket, uint8_t* buffer, size_t sz){
    thread thr[SOCKET_NUMBER];
    int block_size = (int)ceil(sz * 1.0 / SOCKET_NUMBER);
    if (block_size == 0)
        block_size = sz;
    for (int i=0; i < SOCKET_NUMBER; i++){
        thr[i] = thread(SendSingular, *(socket + i), buffer + (i * block_size), block_size);
    }
    for (int i=0; i < SOCKET_NUMBER; i++){
        thr[i].join();
    }
    bytes_sent+=sz;
}

void OpenHelper(string haddress, uint16_t hport, int *socket_p0, int *socket_p1) {
    for (int i=0;i< 2 * SOCKET_NUMBER; i++){
        int new_socket = OpenSocket(haddress, hport + i);
        uint8_t buffer[1];
        int rd = 0;
        while (rd != 1)
            rd = read(new_socket,buffer,1);
        if (buffer[0] == 2)
            printf("P%d is connected on port %d \n", i / SOCKET_NUMBER, hport + i);
        else{
            printf("P%d did not connect to helper on port %d \n", i / SOCKET_NUMBER, hport + i);
            exit(-1);
        }

        buffer[0] = 2;
        rd = 0;
        while (rd != 1)
            rd = write(new_socket , buffer , 1);
        if (i < SOCKET_NUMBER)
            socket_p0[i] = new_socket;
        else
            socket_p1[i - SOCKET_NUMBER] = new_socket;
    }
}

void ConnectToHelper(const string address, uint16_t port, int *socket_helper) {
    for (int i=0; i < SOCKET_NUMBER; i++){
        socket_helper[i] = ConnectToSocket(address, port + i);
        uint8_t buffer[1];
        buffer[0] = 2;
        int rd = 0;
        while (rd != 1)
            rd = write(socket_helper[i], buffer, 1);
        rd = 0;
        while (rd != 1)
            rd = read( socket_helper[i] , buffer, 1);
        if (buffer[0] == 2)
            printf("Helper is connected on port %d \n", port+i);
        else{
            printf("Helper did not connect on port %d\n", port+i);
            exit(-1);
        }
    }
}

void OpenP0(string address, uint16_t port, int *socket_p1){
    for (int i=0; i < SOCKET_NUMBER; i++){
        socket_p1[i] = OpenSocket(address, port + i);
        uint8_t buffer[1];
        int rd = 0;
        while (rd != 1)
            rd = read(socket_p1[i] , buffer, 1);
        if (buffer[0] == 1)
            printf("proxy1 is connected on port %d\n", port+i);
        else{
            printf("proxy1 did not connect to P0 on port %d\n", port+i);
            exit(-1);
        }
        buffer[0] = 0;
        rd = 0;
        while (rd != 1)
            rd = write(socket_p1[i] , buffer ,1);
    }
}

void ConnectToP0(const string address, uint16_t port, int *socket_p0){
    for (int i=0; i < SOCKET_NUMBER; i++){
        socket_p0[i] = ConnectToSocket(address, port + i);
        uint8_t buffer[1];
        buffer[0] = 1;
        int rd = 0;
        while (rd != 1)
            rd = write(socket_p0[i] , buffer ,1);
        rd = 0;
        while (rd != 1)
            rd = read(socket_p0[i] , buffer, 1);
        if (buffer[0] == 0)
            printf("P0 is connected on port %d\n", port+i);
        else{
            printf("P0 did not connect to proxy1 on port %d\n", port+i);
            exit(-1);
        }
    }
}

void CloseSocket(int *sock){
    for (int i=0; i < SOCKET_NUMBER; i++){
        close(sock[i]);
    }
}

void PrintBytes(){
    cout << "Bytes Sent:\t" << bytes_sent << endl;
    cout << "Bytes Received:\t" << bytes_received << endl;
    cout<<"Receive Time\t"<<fixed
        << receive_time << setprecision(9) << " sec" << endl;
    cout<<"Send Time\t"<<fixed
        << send_time << setprecision(9) << " sec" << endl;
}

#endif //PML_CONNECTION_H

